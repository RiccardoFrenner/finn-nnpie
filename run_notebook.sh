export PYTHONPATH="${PYTHONPATH}:./src"

notebook=$1

jupyter nbconvert --execute --to notebook --inplace --clear-output --ExecutePreprocessor.timeout=-1 --allow-errors $notebook 2> data/logs/nb_errors.txt
# jupyter nbconvert --execute --to notebook --inplace --clear-output --ExecutePreprocessor.timeout=-1 $notebook 2> logs/nb_errors.txt

# not working
# jupyter run $notebook

# working but does not update notebook
# ./env/myipython -c "%run $notebook"