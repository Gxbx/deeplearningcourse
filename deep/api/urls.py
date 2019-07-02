from django.conf.urls import url, include
from django.urls import path
from rest_framework import routers
from api.viewsets import deepModelViewSet
from api.viewsets import PredictAPIView
router = routers.DefaultRouter()
router.register(r'deep', deepModelViewSet)


urlpatterns = [
    url(r'^', include(router.urls)),
    path('predict/', PredictAPIView.as_view())
]