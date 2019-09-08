# mr_nlp

## Workflow
### First do this on the terminal
`git pull origin master` to pull changes from oracle.\
`git branch feature_branch` to make a new branch to work on.\
`git checkout feature_branch` to move to it.\

Go and make some changes...\
Do the normal local git workflow on your feature branch:\
`git add <some_files>`\
`git commit -m <a nice descriptive message>`\
Use `git push origin master` to publish your feature branch to github.\
If you had merged with your local master here to would get an error when you tried to push,
since commiting to the origin master is restricted.

### Now do this on github (can be done in terminal but more fiddly)
Go to "pull requests", and create a new pull request.\
Wait for the other person to review changes.\
Merge the changes, I prefer to use rebase.\
Use the github button to remove the feature branch from the origin repo.\
Alternatively you can do this in terminal with `git push origin --delete <feature_branch>`

### Back in the terminal, update your own repo
`git checkout master`
`git branch -d feauture_branch`
`git pull origin master`

And you're done.

# Remote working

Any server style process, such a jupyter notebook or tensorboard, that runs on the remote host can be accessed from a local web browser using port forwarding. To do this you just need to specify which ports to forward and where when you ssh into the remote host.

For example, if you have a jupyter notebook running on port 8888 and tensorboard running on 6006, and you want to access them on ports 5001 and 5002, your SSH command becomes:

```ssh -i <path to key> -L 5001:localhost:888 -L 5002:localhost:6006 <username>@<ec2-amazonaws.com etc>```

If you want your connection to do more than just accept outward data from the host, I think you need to enable a custom TCP rule in the instance security settings (very straightforward).

For code editig remotely in VSCode, install the remote development pack, and modify one of your config files in this style:

```
Host ec2-34-244-39-138.eu-west-1.compute.amazonaws.com
    User ubuntu
    HostName ec2-34-244-39-138.eu-west-1.compute.amazonaws.com
    IdentityFile /home/t/Documents/genei/laksh_genei_key.pem
```

Connecting to the remote host in vscode will then allow you to do all your editing as if you were running vscode on the local machine.
