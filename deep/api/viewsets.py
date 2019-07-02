from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.filters import SearchFilter
from api.serializers import deepModelSerializer
from api.models import deepModel
import sys, os
sys.path.append(os.getcwd())

import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))


class deepModelViewSet(viewsets.ModelViewSet):
    queryset = deepModel.objects.all()
    filter_backends = [SearchFilter]
    search_fields = ['=name']
    serializer_class = deepModelSerializer
from keras.models import load_model




class PredictAPIView (APIView):
    def post(self, request):
        loaded_model = load_model('E:/Development/deeplearning/deep/api/save/models')
        loaded_model.load_weights('E:/Development/deeplearning/deep/api/save/weights/weights.epoch.hdf5')
        deeModelObjs = {
            'MLP' : loaded_model,
            'LSTM' : loaded_model,
            'AutoEncoder' : loaded_model
        }
        #El uso del modelo.
        model = deeModelObjs('MLP')
        value = request.data.get('Test')
        try:
            predictions = model.predict(value)
        except Exception as error:
            return Response({'message': str(error)}, 
                status=status.HTTP_400_BAD_REQUEST)
        return Response({'success': predictions})
