# ZZBktUtils

### 使用步骤
* 安装`pip install --user --upgrade ZhuanZhuanBktUtil`

* 导入包 `from ZhuanZhuanBktUtil import util`
* 准备参数

`data_path="./bkt10w.csv",
label="click_label",
n_buckets=7,
epochs=2,
columns="col1,col2,col3",
`
* 运行run方法：util.run()

`
util.run(data_path=data_path, label=label,n_buckets=n_buckets ,epochs=epochs, columns=columns)
`
### run方法参数详解
* 
