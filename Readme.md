### Introducation
This package provide easy to use functions that translate clickstream data into useful features based on learning theory.

Two major features are implemented:
1. regularity
2. procrastination

Following indicators are available to measure regularity:

|indicator|meaning|dimension|  
|---|---|---|
|pdh|peak on day hour|intra-day|
|pwd|peak on week day|intra-week|
|fwh|Periodicity of week hour|intra-week|
|fwd|Periodicity of week day|intra-week|
|fdh|Periodicity of day hour|intra-day|
|ws1|Weeks similarity measure 1|intra-week|
|ws2|Weeks similarity measure 2|intra-week|
|ws3|Weeks similarity measure 3|intra-week|

Most of the time-based regularity comes from the work [Boroujeni, M. S., Sharma, K., Kidziński, Ł., Lucignano, L., & Dillenbourg, P. (2016, September). How to quantify student’s regularity?. In European Conference on Technology Enhanced Learning (pp. 277-291). Springer, Cham.](https://link.springer.com/chapter/10.1007/978-3-319-45153-4_21). You could find detail definition of above indicators there.

Following indicators are availabel to measure procrastination
+ WeightedMean

### Installation
```bash
git clone https://github.com/banchisme/clickfeatures.git
pip install clickfeatures/
```

### Quick Start
detailed examples are in `doc/regularity_examples.ipynb` and `doc/procrastination_examples.ipynb`


### Todo
1. better explain `timestamp.TimeStamps` in `doc/procrastination_examples.ipynb`
2. better documentation in the functions
3. write an example for activity regularity