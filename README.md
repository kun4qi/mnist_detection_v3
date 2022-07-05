#mnistの1を正常画像、0を異常画像
train.pyで学習させてresultフォルダにモデルを保存、
predict.pyで0と1の画像を含んだtestデータに対して、異常度を計算し、aucrocかaccuracyのスコアを表示]

変更点
train.pyのvalid_stepとtest_stepのlogの部分を変更
