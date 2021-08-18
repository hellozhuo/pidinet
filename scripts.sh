
############### Table 5, Baseline 
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet --config baseline --sa --dil --resume --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/table5_baseline --datadir /path/to/BSDS500 --dataset BSDS #--evaluate /path/to/table5_baseline.tar

# generate maps (no need to convert because baseline is already a vanilla cnn)
python main.py --model pidinet --config baseline --sa --dil -j 4 --gpu 0 --savedir /path/to/table5_baseline --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_baseline.tar

# 101 FPS
python throughput.py --model pidinet --config baseline --sa --dil -j 0 --gpu 0 --datadir /path/to/BSDS500 --dataset BSDS



############### Table 5, PiDiNet
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet --config carv4 --sa --dil --resume --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/table5_pidinet --datadir /path/to/BSDS500 --dataset BSDS #--evaluate /path/to/table5_pidinet.tar

# generate maps with converted pidinet
python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/table5_pidinet --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_pidinet.tar --evaluate-converted

# 96 FPS
python throughput.py --model pidinet_converted --config carv4 --sa --dil -j 0 --gpu 0 --datadir /path/to/BSDS500 --dataset BSDS



############### Table 5, PiDiNet-L
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet --config carv4 --resume --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/table5_pidinet-l --datadir /path/to/BSDS500 --dataset BSDS #--evaluate /path/to/table5_pidinet-l.tar

# generate maps with converted pidinet
python main.py --model pidinet_converted --config carv4  -j 4 --gpu 0 --savedir /path/to/table5_pidinet-l --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_pidinet-l.tar --evaluate-converted

# 135 FPS
python throughput.py --model pidinet_converted --config carv4 -j 0 --gpu 0 --datadir /path/to/BSDS500 --dataset BSDS



############### Table 5, PiDiNet-small
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet_small --config carv4 --sa --dil --resume --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/table5_pidinet-small --datadir /path/to/BSDS500 --dataset BSDS #--evaluate /path/to/table5_pidinet-small.tar

# generate maps with converted pidinet
python main.py --model pidinet_small_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/table5_pidinet-small --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_pidinet-small.tar --evaluate-converted

# 161 FPS
python throughput.py --model pidinet_small_converted --sa --dil --config carv4 -j 2 --gpu 0 --datadir /path/to/BSDS500 --dataset BSDS



############### Table 5, PiDiNet-small-l
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet_small --config carv4 --resume --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/table5_pidinet-small-l --datadir /path/to/BSDS500 --dataset BSDS #--evaluate /path/to/table5_pidinet-small-l.tar

# generate maps with converted pidinet
python main.py --model pidinet_small_converted --config carv4 -j 4 --gpu 0 --savedir /path/to/table5_pidinet-small-l --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_pidinet-small-l.tar --evaluate-converted

# 225 FPS
python throughput.py --model pidinet_small_converted --config carv4 -j 2 --gpu 0 --datadir /path/to/BSDS500 --dataset BSDS



############### Table 5, PiDiNet-tiny
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet_tiny --config carv4 --sa --dil --resume --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/table5_pidinet-tiny --datadir /path/to/BSDS500 --dataset BSDS #--evaluate /path/to/table5_pidinet-tiny.tar

# generate maps with converted pidinet
python main.py --model pidinet_tiny_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/table5_pidinet-tiny --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_pidinet-tiny.tar --evaluate-converted

# 182 FPS 
python throughput.py --model pidinet_tiny_converted --sa --dil --config carv4 -j 2 --gpu 0 --datadir /path/to/BSDS500 --dataset BSDS



############### Table 5, PiDiNet-tiny-l
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet_tiny --config carv4 --resume --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/table5_pidinet-tiny-l --datadir /path/to/BSDS500 --dataset BSDS #--evaluate /path/to/table5_pidinet-tiny-l.tar

# generate maps with converted pidinet
python main.py --model pidinet_tiny_converted --config carv4 -j 4 --gpu 0 --savedir /path/to/table5_pidinet-tiny-l --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_pidinet-tiny-l.tar --evaluate-converted

# 253 FPS
python throughput.py --model pidinet_tiny_converted --config carv4 -j 2 --gpu 0 --datadir /path/to/BSDS500 --dataset BSDS



############### Table 6, PiDiNet
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet --config carv4 --sa --dil --resume --iter-size 24 -j 4 --gpu 0 --epochs 14 --lr 0.005 --lr-type multistep --lr-steps 8-12 --wd 1e-4 --savedir /path/to/table6_pidinet --datadir /path/to/NYUD --dataset NYUD-image --lmbda 1.3 #--evaluate /path/to/table6_pidinet.tar

# generate maps with converted pidinet
python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/table6_pidinet --datadir /path/to/NYUD --dataset NYUD-image --lmbda 1.3 --evaluate /path/to/table6_pidinet.tar --evaluate-converted

# 66 FPS
python throughput.py --model pidinet_converted --sa --dil --config carv4 -j 1 --gpu 0 --datadir /path/to/NYUD --dataset NYUD-image



############### Table 7, PiDiNet
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet --config carv4 --sa --dil --resume --iter-size 24 -j 4 --gpu 0 --epochs 14 --lr 0.005 --lr-type multistep --lr-steps 8-12 --wd 1e-4 --savedir /path/to/table7_pidinet --datadir /path/to/Multicue/multicue_v2 --dataset Multicue-boundary-1 #--evaluate /path/to/table7_pidinet.tar

# generate maps with converted pidinet
python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/table7_pidinet --datadir /path/to/Multicue/multicue_v2 --dataset Multicue-boundary-1 --evaluate /path/to/table7_pidinet.tar --evaluate-converted

# 17 FPS
python throughput.py --model pidinet_converted --sa --dil --config carv4 -j 1 --gpu 0 --datadir /path/to/Multicue/multicue_v2 --dataset Multicue-boundary-1

