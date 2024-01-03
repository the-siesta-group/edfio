# Changelog

## [Unreleased]

### Added
- Make EDF+ header fields `patient` and `recording` more tolerant regarding non-compliant inputs: omitted subfields are returned as `X` instead of throwing an exception ([#18](https://github.com/the-siesta-group/edfio/pull/18)).

### Fixed
- Allow reading more non-standard starttime and startdate fields ([#30](https://github.com/the-siesta-group/edfio/pull/30)).

## [0.4.0] - 2023-12-22

### Changed
- Exclude EDF+ annotation signals from `Edf.signals`, `Edf.num_signals`, and `Edf.drop_signals()` ([#25](https://github.com/the-siesta-group/edfio/pull/25)).
- Provide more concise `__repr__` for `Edf` and `EdfSignal` ([#26](https://github.com/the-siesta-group/edfio/pull/26)).
- `Edf.append_signals()` now inserts new signals after the last ordinary (i.e. non-annotation) signal ([#29](https://github.com/the-siesta-group/edfio/pull/29)).

### Added
- Expand `~` (tilde) to the user's home directory in `edf_file` argument of `read_edf()` and `target` argument of `Edf.write()` ([#23](https://github.com/the-siesta-group/edfio/pull/23)).

### Fixed
- Avoid floating point errors sometimes preventing the creation of an Edf with signals that are actually compatible in duration ([#15](https://github.com/the-siesta-group/edfio/pull/15)).
- Allow reading EDF+ startdate and birthdate with non-uppercase month ([#19](https://github.com/the-siesta-group/edfio/pull/19)).
- Disallow setting signal label to `"EDF Annotations"` ([#28](https://github.com/the-siesta-group/edfio/pull/28)).

## [0.3.1] - 2023-12-06

### Added
- Add optional parameter `sampling_frequency` to `EdfSignal.update_data()` for changing the sampling frequency of the signal when updating its data ([#13](https://github.com/the-siesta-group/edfio/pull/13)).
- Add `update_data_record_duration` method to class `Edf` for updating the `data_record_duration` field of the EDF header ([#14](https://github.com/the-siesta-group/edfio/pull/14)).

## [0.3.0] - 2023-11-30

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
