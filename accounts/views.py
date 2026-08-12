from django.contrib.auth import login
from django.shortcuts import redirect, render

from .forms import InstructorSignUpForm


def signup(request):
    if request.user.is_authenticated:
        return redirect("courses:dashboard")

    if request.method == "POST":
        form = InstructorSignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("courses:dashboard")
    else:
        form = InstructorSignUpForm()

    return render(request, "accounts/signup.html", {"form": form})
