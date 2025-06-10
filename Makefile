#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y taxifare || :
	@pip install -e .


run_api:
	uvicorn api.fast:app --reload
