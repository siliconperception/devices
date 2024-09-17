default:
	echo "make convnext imagenet"

imagenet:
	python imagenet.py --lr 0.0001 --train --test --roi 768 --batch 10 --opt adamw --weight_decay 0.01 --freeze

convnext:
	python convnext.py --loss hyb --show --lr 0.0001 --opt adamw --weight_decay 0.01 --train --batch 10 --save ie-convnext-scratch.pt --avg 1000 --centercrop --nsamples 50000 --slide 100 --sched plat --step 10000 --factor 0.9 --saveinterval 100000
