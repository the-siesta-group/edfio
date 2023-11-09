{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {%- if methods != ["__init__", "count", "index"] %}
   :members:

   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}
   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}

   {%- if methods != ["__init__"] %}
   .. rubric:: {{ _('Methods') }}
   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endif %}
