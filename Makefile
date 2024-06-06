default:
	echo "Usage: make {ie100-050-240, ie100-200-60, ie100-500-24}"

ie100-050-240:
	python ie100.py --model ie100-050-240 --nbatch 1
	python ie100.py --model ie100-050-240 --show

ie100-200-060:
	python ie100.py --model ie100-200-060 --nbatch 1
	python ie100.py --model ie100-200-060 --show

ie100-500-024:
	python ie100.py --model ie100-500-024 --nbatch 1
	python ie100.py --model ie100-500-024 --show
clean:
	rm *.pt
