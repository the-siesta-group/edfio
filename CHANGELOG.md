# Changelog

## [Unreleased]

### Changed
- When `EdfSignal.physical_min` or `EdfSignal.physical_max` do not fit into their header fields, they are now always rounded down or up, respectively, to ensure all physical values lie within the physical range ([#2](https://github.com/the-siesta-group/edfio/pull/2)).
- Support non-standard header fields (not encoded as UTF-8) by replacing incompatible characters with "�" ([#4](https://github.com/the-siesta-group/edfio/pull/4)).


## [0.1.0] - 2023-11-09

Initial release 🎉

### Added
- Support for reading and writing EDF and EDF+C files.
- I/O from/to files and file-like objects.
- `Edf.slice_between_seconds()` and `Edf.slice_between_annotations()` for slicing a recording in time.
- `Edf.drop_annotations()` for dropping EDF+ annotations based on their text.
- `Edf.drop_signals()` for dropping individual EDF signals by specifying their index or label.
- `Edf.anonymize()` for anonymizing all identifying header fields.
