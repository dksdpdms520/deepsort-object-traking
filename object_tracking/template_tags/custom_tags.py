from django.template.defaulttags import register


@register.filter
def get_item(dictionary, key):
    return dictionary[key]


@register.filter
def add_number(value, num):
    return int(value) + num