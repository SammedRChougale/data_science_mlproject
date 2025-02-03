"""
Elastic Beanstalk:
        ""creating python.config file to tell elastic beanstalk instance saying 'what is entry point of the application?'

1. Upload your code- 
        You deploy your application(e.g., Python, Django, Flask, Java,
                                    Node.js, etc.)
2. Elastic Beanstalk Provisions Resources - It automatically creates:
    1. EC2 instances to run your app
    2. Load balancer (if needed)
    3. Auto-scaling group to handle traffic
    4. RDS database (optional).
3. Monitor and Scale - AWS handle scaling, monitoring, and logging.



Example:

Steps to Deploy a Django App on AWS Elastic Beanstalk

1. Install AWS CLI & Elastic Beanstalk CLI
First, install the AWS Elastic Beanstalk (EB) CLI:

```pip install awsebcli --upgrade
```

Check the installation:

```eb --version
```

2. Configure AWS Credentials
Run:

```
aws configure
```

Enter:
    1.AWS Access Key ID
    2.AWS Secret Access Key
    3.Region (e.g., us-east-1)


3. Initialize Elastic Beanstalk
Inside your Django project directory, run:
```
eb init
```
    1.Choose the application name.
    2.Select the platform (Python).
    3.Set up SSH access if needed.

4. Create an Elastic Beanstalk Environment

Run:
```
eb create my-env
```

    This will:
    1.Create an EC2 instance.
    2.Configure Elastic Load Balancer (ELB).
    3.Set up Auto Scaling.

5. Deploy the Django Application
Run:

```eb deploy
```
This uploads your Django app to AWS.

6. View Your Application
Check running environments:

```
eb status
```

Open the app in a browser:
```
eb open
```


Additional Configurations
Handling Database (RDS)
If your Django app needs a database, add Amazon RDS:
    1. Create an RDS instance in AWS.
    2. Update Django settings (settings.py):

python
Copy
```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'your_db_name',
        'USER': 'your_db_user',
        'PASSWORD': 'your_db_password',
        'HOST': 'your-rds-endpoint',
        'PORT': '5432',
    }
}
```
Handling Static & Media Files (S3)
    1.Store static files in Amazon S3 to improve performance.
    2.Use django-storages:
    bash
    Copy
    ```pip install boto3 django-storages
    ```
    3.Update settings.py:
    python
    Copy
    ```
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    STATICFILES_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    ```

Advantages of AWS Elastic Beanstalk
✅ Fully Managed – No need to configure servers manually.
✅ Scalable – Auto-scaling adjusts to traffic spikes.
✅ Supports Multiple Languages – Python, Java, Node.js, PHP, Ruby, Go, etc.
✅ Easy Deployment – Simple eb deploy command.

Next Steps
    1. Add SSL (HTTPS) using AWS ACM.
    2. Enable logging & monitoring using CloudWatch.
    3. Use Amazon S3 for static/media files.
    4. Set up custom domains with Route 53.

"""