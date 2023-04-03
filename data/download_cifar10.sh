echo "Downloading CIFAR-10..."

cd data  # assume we are running this from the root folder

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz;    # 16 MB (md5: c58f30108f718f92721af3b95e74349a)
tar -xf cifar-10-python.tar.gz;

cd ..

echo "done!"
