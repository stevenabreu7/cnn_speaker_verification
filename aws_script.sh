git clone https://github.com/stevenabreu7/cnn_speaker_verification.git
cd cnn_speaker_verification
pip install awscli
awscli configure
mkdir dataset
cd dataset
aws s3 cp s3://11785fall2018/hw2p2_A.tar.gz .
aws s3 cp s3://11785fall2018/hw2p2_B.tar.gz .
aws s3 cp s3://11785fall2018/hw2p2_C.tar.gz .
tar -xzf hw2p2_A.tar.gz
rm hw2p2_A.tar.gz
tar -xzf hw2p2_B.tar.gz
rm hw2p2_B.tar.gz
tar -xzf hw2p2_C.tar.gz
rm hw2p2_C.tar.gz
cd .. 
source activate pytorch_p27
python preprocess.py dataset 1
python preprocess.py dataset 2
python preprocess.py dataset 3
python preprocess.py dataset 4
python preprocess.py dataset 5
python preprocess.py dataset 6
python preprocess.py dataset dev
python preprocess.py dataset test
cd dataset
mkdir _raw
mv *.npy _raw 
mv _raw/*.preprocessed.npz .
cd ..
mkdir models
source deactivate
source activate pytorch_p36
python training.py resnet_full --maxlen 14000 --bsize 16 --parts 6 --epochs 50