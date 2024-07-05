default:
	echo "Usage: make pretrain test"

scratch::
	python pretrain.py --debug --encoder ie120-050-240 --lr 0.0005 --batch 30 --save --scratch --show

pretrain::
	python pretrain.py --debug --encoder ie120-050-240 --lr 0.0005 --batch 20 --resize 0.1 --alt alt1 --save --train 3000000 --scratch
	#python pretrain.py --debug --encoder ie120-050-240 --lr 0.0005 --batch 10 --alt alt1 --save --train 300000 --resize --scale 0.1 --single 0.50
	#python pretrain.py --debug --encoder ie120-050-240 --lr 0.0001 --batch 20 --alt alt1 --save --train 300000 --resize --scale 0.1 --single 0.50

finetune::
	python finetune.py --debug --encoder ie120-050-240.pt --freeze --lr 0.0005 --batch 50 --save --train 120000

test::
	python test.py --alt alt1 --debug --quad

test2::
	python test.py --alt alt2 --model finetune.pt --debug --single --finetune

clean:
	rm *.pt
