default:
	echo "Usage: make {ie120-050-240, ie120-200-60, ie120-500-24}"

ie120-050-240:
	python ie120.py --model ie120-050-240 --nbatch 1
	python ie120.py --model ie120-050-240 --show

ie120-200-060:
	python ie120.py --model ie120-200-060 --nbatch 1
	python ie120.py --model ie120-200-060 --show

ie120-500-024:
	python ie120.py --model ie120-500-024 --nbatch 1
	python ie120.py --model ie120-500-024 --show
clean:
	rm *.pt
