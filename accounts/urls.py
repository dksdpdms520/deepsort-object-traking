from django.contrib.auth import views as auth_views
from django.urls import path

from . import views
from .forms import LoginForm

app_name = 'accounts'

urlpatterns = [
    path('register/', views.account_register, name='account_register'),
    path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html',
                                                form_class=LoginForm), name='account_login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/accounts/login/'), name='account_logout'),
]
