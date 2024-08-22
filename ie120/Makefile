default:
	echo "Usage: make dataset pretrain finetune test"

dataset::
	python dataset.py --debug

pretrain::
	python pretrain.py --debug --centercrop --resize 0.05 --x1 --x2 --x3 --batch 32 --lr 0.1 --opt sgd --save ie120-sgd-scratch.pt --sched plat --step 1000 --factor 0.7 --avg 1000 --scratch

finetune::
	python finetune.py --debug --encoder ie120-sgd-scratch.pt --lr 0.1 --batch 32 --sched plat --step 1000 --avg 1000 --factor 0.7 --freeze --save finetune.pt 

test::
	python test.py --debug --model finetune.pt

coco_train::
	python coco.py --lr 0.1 --opt sgd --batch 20 --ann ../coco/annotations/instances_train2017.json --img ../coco/train2017 --show --train --sched plat --step 200 --avg 200 --factor 0.7 --freeze

coco_test::
	python coco.py --ann ../coco/annotations/instances_val2017.json --img ../coco/val2017 --decoder decoder.pt --freeze  --test

alt2::
	python pretrain.py --debug --centercrop --resize 0.05 --x1 --x2 --x3 --batch 32 --lr 0.1 --opt sgd --save ie-alt2-scratch.pt --sched plat --step 1000 --factor 0.7 --avg 1000 --alt alt2 --train --device cuda

imagenet::
	python imagenet.py --batch 32 --lr 0.1 --opt sgd --save imagenet.pt --sched plat --step 1000 --factor 0.7 --avg 1000 --encoder ie120-sgd-scratch.pt --alt alt1 --train --test --device cuda --freeze

