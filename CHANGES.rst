Changes
=======

0.2 (2015-12-11)
----------------

* **backwards-incompatible**: ``crf.tagger`` attribute is renamed to
  ``crf.tagger_``; when model is not trained accessing this attribute
  no longer raises an exception, its value is set to None instead.

* new CRF attributes available after training:

    * ``classes_``
    * ``size_``
    * ``num_attributes_``
    * ``attributes_``
    * ``state_features_``
    * ``transition_features_``

* Tutorial is added.

0.1 (2015-11-27)
----------------

Initial release.
