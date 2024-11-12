default:
	echo "make pretrain finetune"

pretrain_encoder:
	python resnet.py --opt adamw --weight_decay 0.0 --mode pretrain --workers 12 --batch 20 --save ie120r_res18.pt --show --verbose --loss pdist --sched cos --lr 0.001 --step 100 --eta_min 0.00001 --tmax 100 --alt medium

finetune_encoder:
	python resnet.py --lr 0.00001 --mode finetune_encoder --encoder ie120r_res18.pt --workers 10 --batch 20 --verbose --nbatch 10000

finetune_pretrained:
	python resnet.py --lr 0.00001 --mode finetune_encoder --pretrained --workers 10 --batch 20 --verbose --nbatch 10000

finetune_resnet18:
	time python resnet.py --lr 0.00001 --mode finetune_resnet18 --workers 10 --batch 20 --verbose --nbatch 10000

