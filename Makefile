install:
	pip install -r requirements.txt

run:
	python src/model/knn.py
	python src/model/xgboostModel.py
	python src/model/random_forest.py
	python src/model/neural_network.py
	python src/model/linearRegression.py
	python src/model/linearSVM.py
	python src/model/lasso.py
	python src/model/ridge.py
	python src/model/comparison.py

test:
	python -m pytest tests/ -v