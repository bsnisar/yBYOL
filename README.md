
## Floyd (update data)

#### Updating/Versioning Your Dataset
If you've made changes to your dataset and would like to upload it again, use the following steps. You'll notice they are the same as uploading your dataset the first time:

cd into your dataset's directory
Run `floyd data init <dataset_name>` to prepare to upload
Run `floyd data upload`
Your dataset will be versioned for you, so you can still reference the old one if you'd like. Datasets will be named with sequential numbers, like this:
```
alice/datasets/foo/1
alice/datasets/foo/2
alice/datasets/foo/3
...
