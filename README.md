# 20160806mps_yokohama

お疲れ様です。
ここに置いてあるファイルは金子さんのオリジナルを with Python 2.7 conpatiblility で down conversion したものです。

## MNIST データ次元について
784次元です。手書き文字の生データは縦 32 ドット，横32ドットを切り出した28ドット×28ドットです。
従って一文字のデータは 784次元のベクトルです。

## 初期化について

Caffe などでは Xavier 初期化などが使われます。とくにハイパーボリックタンジェントでは
論文としては "Understanding the difficulty of training deep feedforward neural networks" Glorot and Bengio(2010) を参照してください。
