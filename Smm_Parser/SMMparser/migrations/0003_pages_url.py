# Generated by Django 4.2.7 on 2023-11-15 12:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SMMparser', '0002_pages_categ4_alter_pages_categ_root_alter_pages_id_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='pages',
            name='url',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
