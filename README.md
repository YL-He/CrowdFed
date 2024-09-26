# CrowdFed
This is the code repository for the paper under review.

### Setup

The project requires a python version higher than 3.8, the pytorch 2.1 and tensorboard 2.12.

Please execute the following commands before running:

```
makedir data
  makedir Results
```

### How to run

You can run the default as follows：

```python
python main.py
```

You can set match to 1 to use the category-matching strategy:

```python
python main.py --match 1
```

### How to view results directly

* Please execute the following command after runnig:

```
tensorboard --logdir=runs/ --bind_all
```

* Open your browser and visit `http://127.0.0.1:6006/`
