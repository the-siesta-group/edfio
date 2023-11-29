# Changelog

## [Unreleased]
### Changed
- `Edf.signals` can not be set anymore. Use `Edf.append_signals()`, `Edf.drop_signals()`, and `EdfSignal.update_data()` instead ([#10](https://github.com/the-siesta-group/edfio/pull/10)).

### Added
- Allow creating a new Edf containing only annotations ([#7](https://github.com/the-siesta-group/edfio/pull/7)).
- Add `EdfSignal.update_data` for overwriting the physical values with an array of equal length ([#10](https://github.com/the-siesta-group/edfio/pull/10)).
- Allow adding new signals to an Edf with `Edf.append_signals` ([#10](https://github.com/the-siesta-group/edfio/pull/10)).

### Fixed
- Disallow creating a new Edf where local patient/recording identification subfields are empty strings ([#6](https://github.com/the-siesta-group/edfio/pull/6)).
- Allow retrieving the starttime from a file where the reserved field indicates it is an EDF+C but no annotations signal is present ([#8](https://github.com/the-siesta-group/edfio/pull/8)).
- Disallow removing the EDF+ timekeeping signal with `Edf.drop_signals` ([#10](https://github.com/the-siesta-group/edfio/pull/10)).

## [0.2.1] - 2023-11-17

### Fixed
- Disallow creating a new Edf with a recording startdate not between 1985-01-01 and 2084-12-31 ([#5](https://github.com/the-siesta-group/edfio/pull/5)).

## [0.2.0] - 2023-11-16

### Added
- Support non-standard header fields (not encoded as UTF-8) by replacing incompatible characters with "�" ([#4](https://github.com/the-siesta-group/edfio/pull/4)).

### Fixed
- When `EdfSignal.physical_min` or `EdfSignal.physical_max` do not fit into their header fields, they are now always rounded down or up, respectively, to ensure all physical values lie within the physical range ([#2](https://github.com/the-siesta-group/edfio/pull/2)).
- The calculation of `num_data_records` from signal duration and `data_record_duration` is now more robust to floating point errors ([#3](https://github.com/the-siesta-group/edfio/pull/3))

## [0.1.1] - 2023-11-09

### Fixed
- Use correct path to sphinx config file in `.readthedocs.yml`

## [0.1.0] - 2023-11-09

Initial release 🎉

### Added
- Support for reading and writing EDF and EDF+C files.
- I/O from/to files and file-like objects.
- `Edf.slice_between_seconds()` and `Edf.slice_between_annotations()` for slicing a recording in time.
- `Edf.drop_annotations()` for dropping EDF+ annotations based on their text.
- `Edf.drop_signals()` for dropping individual EDF signals by specifying their index or label.
- `Edf.anonymize()` for anonymizing all identifying header fields.
