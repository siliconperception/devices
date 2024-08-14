default:
	echo "Usage: make dataset pretrain finetune test"

dataset::
	python dataset.py --debug

pretrain::
	python pretrain.py --debug --centercrop --resize 0.05 --x1 --x2 --x3 --batch 32 --lr 0.1 --opt sgd --save ie120-sgd-scratch --sched plat --step 1000 --factor 0.7 --avg 1000 --scratch

finetune::
	python finetune.py --debug --encoder ie120-sgd-scratch.pt --lr 0.1 --batch 32 --sched plat --step 200 --avg 200 --factor 0.7 --freeze --save finetune.pt 

test::
	python test.py --debug --model finetune.pt
