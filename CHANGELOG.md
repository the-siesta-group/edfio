# Changelog

## [Unreleased]

## [0.1.0] - 2023-11-09

Initial release 🎉

### Added
- Support for reading and writing EDF and EDF+C files.
- I/O from/to files and file-like objects.
- `Edf.slice_between_seconds()` and `Edf.slice_between_annotations()` for slicing a recording in time.
- `Edf.drop_annotations()` for dropping EDF+ annotations based on their text.
- `Edf.drop_signals()` for dropping individual EDF signals by specifying their index or label.
- `Edf.anonymize()` for anonymizing all identifying header fields.
