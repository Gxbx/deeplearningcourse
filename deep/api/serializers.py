from api.models import deepModel
from rest_framework import serializers

class deepModelSerializer (serializers.ModelSerializer):
    class Meta:
        model = deepModel
        fields = ('__all__')
    