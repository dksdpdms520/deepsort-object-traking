from django.contrib import auth
from django.shortcuts import redirect, render

from .forms import RegisterForm


def account_register(request):
    if request.method == 'POST':
        regitseration_form = RegisterForm(request.POST)

        if regitseration_form.is_valid():
            user = regitseration_form.save()
            user.email = regitseration_form.cleaned_data['email']
            user.set_password(regitseration_form.cleaned_data['password'])
            user.save()
            auth.login(request, user)
            return redirect('pages:home')
    else:
        regitseration_form = RegisterForm()

    return render(request, 'accounts/register.html', {'form': regitseration_form})
