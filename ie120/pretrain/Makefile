default:
	echo "Usage: make pretrain test"

scratch::
	python pretrain.py --debug --encoder ie120-050-240 --lr 0.0005 --batch 30 --save --scratch --show

pretrain::
	python pretrain.py --debug --encoder ie120-050-240 --lr 0.0005 --batch 32 --save --show --resize 0.0 --workers 20 --x1 --train 100000 --step 10000 --scratch
	cp ie120-050-240.pt ie120-050-240.pt.CP1
	python pretrain.py --debug --encoder ie120-050-240 --lr 0.0005 --batch 32 --save --show --resize 0.0 --workers 20 --x1 --x2 --train 100000 --step 10000
	cp ie120-050-240.pt ie120-050-240.pt.CP2
	python pretrain.py --debug --encoder ie120-050-240 --lr 0.0005 --batch 32 --save --show --resize 0.0 --workers 20 --x1 --x2 --x3 --train 100000 --step 10000
	cp ie120-050-240.pt ie120-050-240.pt.CP3
	python pretrain.py --centercrop --debug --encoder ie120-050-240 --lr 0.0001 --batch 100 --save --show --resize 0.1 --workers 20 --x1 --x2 --x3 --train 10000000 --step 10000000
	cp ie120-050-240.pt ie120-050-240.pt.CP4

dummy::
	#python pretrain.py --debug --encoder ie120-050-240 --lr 0.0005 --batch 32 --save --show --resize 0.0 --workers 20
	#python pretrain.py --debug --encoder ie120-050-240 --lr 0.0005 --batch 20 --resize 0.1 --alt alt1 --save --train 3000000 --scratch
	#python pretrain.py --debug --encoder ie120-050-240 --lr 0.0005 --batch 10 --alt alt1 --save --train 300000 --resize --scale 0.1 --single 0.50
	#python pretrain.py --debug --encoder ie120-050-240 --lr 0.0001 --batch 20 --alt alt1 --save --train 300000 --resize --scale 0.1 --single 0.50

finetune::
	python finetune.py --centercrop --debug --encoder ie120-050-240.pt --freeze --lr 0.0005 --batch 50 --save --workers 20 --step 1000 --end_factor 0.2

test::
	python oldtest.py --alt alt1 --debug --quad

test2::
	python test.py --model finetune.pt --debug --single --batch 50 --nbatch 1000 --x1 --centercrop
	#python test.py --alt alt2 --model finetune.pt --debug --single --finetune

clean:
	rm *.pt
