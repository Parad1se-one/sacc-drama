# DRAMA Benchmark Package

This package provides a compact benchmark registry for the GitHub submission.

Public benchmark names:

- `easy`
- `hard`

## API

- `DRAMA.list_benchmarks()`
- `DRAMA.get_benchmark(name)`
- `DRAMA.get_config(name)`
- `DRAMA.get_metadata(name)`
- `DRAMA.make_env(name, seed=...)`

## CLI examples

```bash
python -m DRAMA list
python -m DRAMA describe easy
python -m DRAMA describe hard
python -m DRAMA sample hard --steps 3
```
