# 環境構築

## librosa のトラブルシューティング
From here: https://stackoverflow.com/questions/30219360/seewave-install-error-sndfile-h-file-not-found-for-r-3-2-0-under-osx-yose/30980601#30980601
```
On linux it is sufficient to install libsndfile library, for example with
sudo apt-get install libsndfile1-dev
On OS X just do:
brew install libsndfile
```


## Light GBMのインストール
### cmake
```
sudo apt install cmake
```

このサイト[https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#linux]を参照して、
```
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake ..
make -j4
```

もしCXXコンパイラのエラーが出たら、以下を実行後に再度`cmake ..`を実行
```
sudo apt-get update && sudo apt-get install build-essential
```

