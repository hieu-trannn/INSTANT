_base_ = [
    "base/efficientformer-l1.py",
    "../datasets/places.py",
    "../schedules/places.py"
]

model = dict(head=dict(num_classes=365))
