{% assign notation_keys = include.keys | split: ' ' %}

| Symbol | Meaning |
| --- | --- |
{% for row in site.data.notation %}{% if notation_keys contains row.key %}| {{ row.symbol }} | {{ row.meaning }} |
{% endif %}{% endfor %}
