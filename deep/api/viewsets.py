from rest_framework import viewsets
from rest_framework.filters import SearchFilter
from api.serializers import deepModelSerializer
from api.models import deepModel

class deepModelViewSet(viewsets.ModelViewSet):
    queryset = deepModel.objects.all()
    filter_backends = [SearchFilter]
    search_fields = ['=name']
    serializer_class = deepModelSerializer