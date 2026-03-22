from django.shortcuts import render

# Create your views here.
# Create your views here.
from django.shortcuts import render,redirect
from django.contrib import messages
from users.models import UserRegistrationModel


# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})



def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', context={'data': data})




def ActivateUsers(request):
    if request.method == 'GET':
        user_id = request.GET.get('uid')
        
        if user_id:  # Ensure user_id is not None
            status = 'activated'
            print("Activating user with ID =", user_id)
            user = UserRegistrationModel.objects.filter(id=user_id)
            if user.exists():
                user.update(status=status)
                messages.success(request, f"User '{user.first().name}' has been activated successfully!")
            else:
                messages.error(request, "User not found.")
        else:
            messages.error(request, "Invalid User ID.")

        # Redirect to the view where users are listed after activation
        return redirect('RegisterUsersView')

def DeleteUsers(request):
    if request.method == 'GET':
        user_id = request.GET.get('uid')
        
        if user_id:  # Ensure user_id is not None
            print("Deleting user with ID =", user_id)
            user = UserRegistrationModel.objects.filter(id=user_id)
            if user.exists():
                name = user.first().name
                user.delete()
                messages.warning(request, f"User '{name}' has been deleted.")
            else:
                messages.error(request, "User not found.")
        else:
            messages.error(request, "Invalid User ID.")
            
        # Redirect to the view where users are listed after deletion
        return redirect('RegisterUsersView')

