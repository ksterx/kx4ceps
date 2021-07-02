# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import requests
import json


def is_notebook():
    """Determine wheather is the environment Jupyter Notebook"""
    if "get_ipython" not in globals():
        # Python shell
        return False
    env_name = get_ipython().__class__.__name__
    if env_name == "TerminalInteractiveShell":
        # IPython shell
        return False
    # Jupyter Notebook
    return True


def line_notify(message):
    token = 'FkhhJG7A4yZmawTo09qPv27sGCcEXWzU1tr03t4xsyJ'
    api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + token}
    requests.post(api, data=payload, headers=headers)


def slack_notify(message, usr='Python'):
    web_hook_url = 'https://hooks.slack.com/services/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'
    requests.post(web_hook_url, data=json.dumps({
        'text': message,
        'username': usr,
        'link_names': 1
    }))


