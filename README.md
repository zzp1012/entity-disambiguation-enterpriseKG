<div style="text-align:center">
	<img src="images/ji_logo.png" alt="Jilogo" style="zoom:60%;" />
</div>
<center>
	<h2>
		VE444 Networks
	</h2>
</center> 
<center>
	<h3>
		Named Disambiguation Problem
	</h3>
</center>
<center>
   <h4>
       FA 2020
    </h4> 
</center>


------------------------------------------

### Abstract

This project is used to help our VE444 Project team work together. 

Once we open source the code for `20FA VE444 term Project` and if you want to refer to our work, please follow the Joint Institute’s honor code and don’t plagiarize these codes directly.

### Repo Structure

```bash
├── data
│   ├── com2com.csv # contains the investment relationships between companies.
│   ├── com_info.csv # contains the basic information about the companies.
│   ├── labels.csv # contains the tuple of investors with same name and a label indicating whether
|   |              # they are the same entity or not
│   └── person2com.csv # contains the investment relationships between companies and investor.
├── final-rpt
│   ├── acmart.cls # the template of final report.
│   ├── ACM-Reference-Format.bst # the template of final report.
│   ├── figures # the directory containing all the figures used in the final report.
│   ├── final-rpt.pdf # the final report of this project
│   ├── main.tex # the tex file of the final report.
│   └── sample.bib # the bibtex file of the final report.
├── images # basically contain some images.
├── milestone-rpt
│   ├── milestone-rpt.md # the source code of milestone-rpt.
│   └── milestone-rpt.pdf # and the pdf version of milestone-rpt
├── poster
│   ├── 444_poster.pdf # the final version of poster
│   └── 444_poster.pptx # the template of the poster
├── prior # under this directory, some prior works are included.
│   ├── code
│   │   ├── company_information_cleaning.ipynb
│   │   ├── data_processing.ipynb
│   │   └── stock_analysis.py
│   ├── README.md
│   └── reference
│       ├── Enterprise-KG-Paper.pdf
│       ├── Enterprise-KG-Slides.pdf
│       ├── graph-database-overview.pdf
│       └── neo4j-algorithms.pdf
├── proposal
│   ├── Proposal.md # the source code of proposal
│   ├── Proposal.pdf # the final pdf version of proposal 
│   └── README.md # the ideas of the projects.
├── README.md # overview of the repo, readme
└── src 
    ├── NED-benchmark.py # the only source code of our project, used to benchmark different methods to
    |                    #solve NED problem
    └── pretrained
        └── node2vec-embedd.txt # the pretrained node embedding vectors using node2vec.
```

### Usage

To benchmark different methods to resolve the NED problem in enterprise KG. Please run following instruction:

```bash
python ./src/NED-benchmark.py --max_itr 1000 --split_ratio 0.7 --seed 10
```

For detailed usage, please check:

```bash
python ./src/NED-benchmark.py --help
```

### Code Style

**Example**:

```python
def sch_random(round_idx, time_counter):
    # set the seed
    np.random.seed(round_idx)

    # random sample clients
    cars = list(channel_data[channel_data['Time'] == time_counter]['Car'])
    client_indexes = []
    if cars:
        client_indexes = list(np.random.choice(cars, max(int(len(cars) / 2), 1), replace=False).ravel())
    logger.info("client_indexes = " + str(client_indexes) + "; time_counter = " + str(time_counter))

    # random local iterations
    local_itr = np.random.randint(2) + 1

    s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
    s_local_itr = str(local_itr)

    return s_client_indexes + "," + s_local_itr
```

1. Plz avoid meaningless combination of letters like `a` or `abc` when naming variables. Name of variable should be meaningful. 
3. Plz add appropriate indentation and blank lines to your code.
4. Plz add enough comments to help others understand your code.
4. If there is any hyper parameter in your code or global parameter, plz use `argparse` or create another file called `config.py` to store them. For example, [config.py](https://github.com/zzp1012/federated-learning-environment/blob/master/fedavg/config.py) and [argparse](https://github.com/zzp1012/federated-learning-environment/blob/master/fedavg/scheduler.py)

### Git Usage

Here are some simple instructions about how to use `Git`.

1. If you want to download the whole project, run following command.

```bash
git clone https://github.com/zzp1012/VE370-Project2.git
```

2. If you want add files to our local git project and remote git project on `github`, run following command.

```bash
# Firstly, plz avoid adding files to master branch on github directly. You can create your own branch locally and remotely.

git branch zzp1012 # create my local branch. Here I name the branch as 'zzp1012'. If you have already created a branch, you can jump to next command.

git checkout zzp1012 # switch to 'zzp1012' branch.

git add * # add all the files to local branch 'zzp1012'.

git commit -m "update" # confirm to add files to local branch 'zzp1012'

git push origin zzp1012 # create branch 'zzp1012' remotely on github and copy your the content on your local branch 'zzp1012' to the remote 'zzp1012'.
```

3. If you want to synchronize files on remote project on `github`, you should run:

```bash
git pull origin master # synchronize files on remote master branch.
git pull origin "you branch name" # the 'master' can be replaced by the name of the other branch created on remote project on github, then you can synchronize files on the specific remote branch.
```

### Reference

[1] Zhu, Y., 2020. *Ve444 Networks Project*.

---------------------------------------------------------------

<center>
    UM-SJTU Joint Institute 交大密西根学院
</center>