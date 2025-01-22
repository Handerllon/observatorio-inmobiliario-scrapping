mkdir lambda_package
pip install -r requirements.txt --target lambda_package
#pip install -r requirements.txt \
#--platform manylinux2014_x86_64 \
#--target=lambda_package \
#--implementation cp \
#--python-version 3.11 \
#--only-binary=:all: --upgrade
