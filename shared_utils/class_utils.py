

def print_instance_attributes(class_instance):
    for attribute, value in class_instance.__dict__.items():
        print(attribute, '=', value)