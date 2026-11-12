"""
Input/Output utilities for loading and saving microscopy data and results.
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional
import logging

import tifffile

try:
    from ..core.data_structures import ImageStack, TrackingResult
except ImportError:
    from core.data_structures import ImageStack, TrackingResult

logger = logging.getLogger(__name__)


def load_image_stack(filename: Union[str, Path], **kwargs) -> ImageStack:
    """
    Load microscopy image stack from file.

    Supports TIFF, HDF5, and other common formats.

    Args:
        filename: Path to image file
        **kwargs: Additional parameters (pixel_size, time_interval, etc.)

    Returns:
        ImageStack object
    """
    filename = Path(filename)
    suffix = filename.suffix.lower()

    if suffix in ['.tif', '.tiff']:
        return _load_tiff(filename, **kwargs)
    elif suffix in ['.h5', '.hdf5']:
        return _load_hdf5(filename, **kwargs)
    elif suffix in ['.npy']:
        return _load_numpy(filename, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_tiff(filename: Path, **kwargs) -> ImageStack:
    """
    Load TIFF stack.

    Pixel size (um) and frame interval (s) come from the keyword arguments if
    given, else from the file (ImageJ resolution + unit + finterval, or OME
    PhysicalSizeX / TimeIncrement), else the ImageStack defaults (0.1 um,
    0.1 s). metadata['scale_source'] records which ('argument', 'file' or
    'default') so results can say where their units came from.
    """
    data = tifffile.imread(filename)

    metadata = {}
    file_scale = {}
    try:
        with tifffile.TiffFile(filename) as tif:
            if tif.imagej_metadata:
                metadata = dict(tif.imagej_metadata)
            file_scale = _scale_from_tiff(tif)
    except Exception as e:  # unreadable tags: fall back to defaults, but say so
        logger.warning(f"Could not read scale from {filename.name}: {e}")

    values, source = {}, {}
    for key, default in (('pixel_size', 0.1), ('time_interval', 0.1)):
        if kwargs.get(key) is not None:
            values[key], source[key] = float(kwargs[key]), 'argument'
        elif file_scale.get(key):
            values[key], source[key] = file_scale[key], 'file'
        else:
            values[key], source[key] = default, 'default'
            logger.warning(f"{filename.name}: no {key} in the file or arguments; using the default {default}")
    metadata['scale_source'] = source

    return ImageStack(
        data=data,
        metadata=metadata,
        pixel_size=values['pixel_size'],
        time_interval=values['time_interval'],
        channels=kwargs.get('channels', 1)
    )


_UM_PER_UNIT = {'micron': 1.0, 'um': 1.0, '\u00b5m': 1.0, '\u03bcm': 1.0, 'nm': 1e-3, 'mm': 1e3}


def _scale_from_tiff(tif) -> Dict:
    """Pixel size (um) and frame interval (s) stored in an ImageJ or OME TIFF, where present."""
    out = {}
    if tif.is_ome and tif.ome_metadata:
        import re
        m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tif.ome_metadata)
        unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', tif.ome_metadata)
        if m:
            out['pixel_size'] = float(m.group(1)) * _UM_PER_UNIT.get(unit.group(1) if unit else 'um', 1.0)
        t = re.search(r'TimeIncrement="([0-9.eE+-]+)"', tif.ome_metadata)
        if t:
            out['time_interval'] = float(t.group(1))
    ij = tif.imagej_metadata or {}
    page = tif.pages[0]
    unit = str(ij.get('unit', '')).strip()
    if 'pixel_size' not in out and unit in _UM_PER_UNIT and 'XResolution' in page.tags:
        num, den = page.tags['XResolution'].value
        if num > 0:
            out['pixel_size'] = den / num * _UM_PER_UNIT[unit]
    if 'time_interval' not in out and ij.get('finterval'):
        out['time_interval'] = float(ij['finterval'])
    return out


def _load_hdf5(filename: Path, **kwargs) -> ImageStack:
    """Load HDF5 file."""
    import h5py

    with h5py.File(filename, 'r') as f:
        # Try to find image data
        if 'data' in f:
            data = f['data'][:]
        elif 'images' in f:
            data = f['images'][:]
        else:
            # Use first dataset
            key = list(f.keys())[0]
            data = f[key][:]

        # Load metadata
        metadata = dict(f.attrs) if hasattr(f, 'attrs') else {}

    return ImageStack(
        data=data,
        metadata=metadata,
        pixel_size=kwargs.get('pixel_size', metadata.get('pixel_size', 0.1)),
        time_interval=kwargs.get('time_interval', metadata.get('time_interval', 0.1)),
        channels=kwargs.get('channels', 1)
    )


def _load_numpy(filename: Path, **kwargs) -> ImageStack:
    """Load numpy array."""
    data = np.load(filename)

    return ImageStack(
        data=data,
        pixel_size=kwargs.get('pixel_size', 0.1),
        time_interval=kwargs.get('time_interval', 0.1),
        channels=kwargs.get('channels', 1)
    )


def save_image_stack(image_stack: ImageStack,
                    filename: Union[str, Path],
                    format: Optional[str] = None):
    """
    Save image stack to file.

    Args:
        image_stack: ImageStack to save
        filename: Output filename
        format: File format (auto-detected from extension if None)
    """
    filename = Path(filename)

    if format is None:
        format = filename.suffix.lower().lstrip('.')

    if format in ['tif', 'tiff']:
        tifffile.imwrite(filename, image_stack.data, imagej=True)
    elif format in ['h5', 'hdf5']:
        import h5py
        with h5py.File(filename, 'w') as f:
            f.create_dataset('data', data=image_stack.data)
            f.attrs['pixel_size'] = image_stack.pixel_size
            f.attrs['time_interval'] = image_stack.time_interval
    elif format == 'npy':
        np.save(filename, image_stack.data)
    else:
        raise ValueError(f"Unsupported format: {format}")


def export_results(tracking_result: TrackingResult,
                   filename: Union[str, Path],
                   format: str = 'csv',
                   include_analysis: bool = False):
    """
    Export tracking results to file.

    Args:
        tracking_result: TrackingResult to export
        filename: Output filename
        format: Export format ('csv', 'hdf5', 'mat', 'xml')
        include_analysis: Include analysis results
    """
    filename = Path(filename)

    if format == 'csv':
        _export_csv(tracking_result, filename, include_analysis)
    elif format == 'hdf5':
        _export_hdf5(tracking_result, filename, include_analysis)
    elif format == 'mat':
        _export_matlab(tracking_result, filename, include_analysis)
    elif format == 'xml':
        _export_xml(tracking_result, filename)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def _export_csv(result: TrackingResult, filename: Path, include_analysis: bool):
    """Export to CSV."""
    df = result.to_dataframe()
    df.to_csv(filename, index=False)

    # Save summary statistics in separate file if requested
    if include_analysis:
        summary_file = filename.with_name(filename.stem + '_summary.csv')

        summary_data = {
            'track_id': [],
            'n_detections': [],
            'duration': [],
            'mean_intensity': []
        }

        for track in result.tracks:
            summary_data['track_id'].append(track.track_id)
            summary_data['n_detections'].append(track.length)
            summary_data['duration'].append(track.duration)
            summary_data['mean_intensity'].append(
                np.mean([d.intensity for d in track.detections])
            )

        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)


def _export_hdf5(result: TrackingResult, filename: Path, include_analysis: bool):
    """Export to HDF5."""
    import h5py

    with h5py.File(filename, 'w') as f:
        # Save tracks as structured array
        df = result.to_dataframe()
        f.create_dataset('tracks', data=df.to_records(index=False))

        # Save metadata
        for key, value in result.metadata.items():
            if isinstance(value, (int, float, str)):
                f.attrs[key] = value


def _export_matlab(result: TrackingResult, filename: Path, include_analysis: bool):
    """Export to MATLAB .mat file."""
    from scipy.io import savemat

    # Convert to MATLAB-friendly format
    tracks_array = []

    for track in result.tracks:
        track_data = {
            'track_id': track.track_id,
            'x': np.array([d.x for d in track.detections]),
            'y': np.array([d.y for d in track.detections]),
            'frame': np.array([d.t for d in track.detections]),
            'intensity': np.array([d.intensity for d in track.detections])
        }

        if track.is_3d:
            track_data['z'] = np.array([d.z for d in track.detections])

        tracks_array.append(track_data)

    savemat(filename, {'tracks': tracks_array})


def _export_xml(result: TrackingResult, filename: Path):
    """Export to TrackMate XML format for compatibility."""
    import xml.etree.ElementTree as ET

    root = ET.Element('TrackMate', version='7.0')

    # Model
    model = ET.SubElement(root, 'Model', spatialunits='pixel', timeunits='frame')

    # All spots
    all_spots = ET.SubElement(model, 'AllSpots', nspots=str(result.n_detections))

    # Organize by frame
    by_frame = {}
    for detection in result.detections:
        if detection.t not in by_frame:
            by_frame[detection.t] = []
        by_frame[detection.t].append(detection)

    for frame, detections in sorted(by_frame.items()):
        frame_elem = ET.SubElement(all_spots, 'SpotsInFrame', frame=str(frame))

        for det in detections:
            spot = ET.SubElement(frame_elem, 'Spot',
                ID=str(id(det)),
                POSITION_X=str(det.x),
                POSITION_Y=str(det.y),
                POSITION_T=str(det.t),
                QUALITY=str(det.snr)
            )
            if det.z is not None:
                spot.set('POSITION_Z', str(det.z))

    # All tracks
    all_tracks = ET.SubElement(model, 'AllTracks')

    for track in result.tracks:
        track_elem = ET.SubElement(all_tracks, 'Track',
            TRACK_ID=str(track.track_id),
            NUMBER_SPOTS=str(track.length),
            NUMBER_GAPS=str(len(track.gap_frames))
        )

        # Edges
        for i in range(len(track.detections) - 1):
            edge = ET.SubElement(track_elem, 'Edge',
                SPOT_SOURCE_ID=str(id(track.detections[i])),
                SPOT_TARGET_ID=str(id(track.detections[i+1])),
                EDGE_TIME=str((track.detections[i].t + track.detections[i+1].t) / 2)
            )

    # Write to file
    tree = ET.ElementTree(root)
    tree.write(filename, encoding='utf-8', xml_declaration=True)
