# Generated by Django 4.2.7 on 2023-11-15 07:43

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('SMMparser', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='pages',
            name='categ4',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pages',
            name='categ_root',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pages',
            name='id',
            field=models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='pages',
            name='name',
            field=models.CharField(max_length=100, unique=True),
        ),
    ]
