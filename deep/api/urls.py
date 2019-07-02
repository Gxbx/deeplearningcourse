from django.conf.urls import url, include
from rest_framework import routers
from api.viewsets import deepModelViewSet

router = routers.DefaultRouter()
router.register(r'deep', deepModelViewSet)

urlpatterns = [
    url(r'^', include(router.urls))
]