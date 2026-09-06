"""Where an annotation came from, and how that is shown.

For this lab a hand-drawn label is ground truth and a SAM 3 mask is a
suggestion waiting to be reviewed. Both live in the same
``{class_name: [annotation, ...]}`` mapping on a frame, so the only thing
that separates them is the ``source`` key the generator wrote. Keeping
those values in one module means the annotation list, the canvas, the
tracking guards and the review filter cannot drift apart from each
other, which is how "is this mine or the model's?" becomes unanswerable.

An annotation with no ``source`` key was drawn by hand. That is the
historical shape of the data — every manual tool has always written a
plain dict — so absence means manual, and no migration is needed for
projects saved before this module existed.
"""

TRACKED_SOURCE = "sam3_track"
DINO_SOURCE = "dino"

#: Sources produced by a model rather than by a person.
AI_SOURCES = frozenset({TRACKED_SOURCE, DINO_SOURCE})

_SOURCE_LABELS = {
    TRACKED_SOURCE: "SAM 3",
    DINO_SOURCE: "DINO",
}


def annotation_source(annotation):
    """Return the raw source string for an annotation, or ``""``."""
    if not isinstance(annotation, dict):
        return ""
    return annotation.get("source") or ""


def is_ai_generated(annotation):
    """True when a model produced this annotation rather than a person."""
    return annotation_source(annotation) in AI_SOURCES


def is_tracked(annotation):
    """True for a mask SAM 3 propagated from another frame."""
    return annotation_source(annotation) == TRACKED_SOURCE


def is_manual(annotation):
    """True when a person drew this annotation."""
    return not is_ai_generated(annotation)


def source_label(annotation):
    """Short display tag for the annotation list, or ``""`` if manual."""
    return _SOURCE_LABELS.get(annotation_source(annotation), "")


def iter_annotations(frame_annotations):
    """Yield ``(class_name, annotation)`` for one frame's mapping."""
    if not frame_annotations:
        return
    for class_name, annotations in frame_annotations.items():
        for annotation in annotations or []:
            yield class_name, annotation


def count_ai_annotations(frame_annotations):
    """How many annotations on one frame came from a model."""
    return sum(
        1
        for _, annotation in iter_annotations(frame_annotations)
        if is_ai_generated(annotation)
    )


def count_manual_annotations(frame_annotations):
    """How many annotations on one frame were drawn by hand."""
    return sum(
        1
        for _, annotation in iter_annotations(frame_annotations)
        if is_manual(annotation)
    )


def frame_has_ai_annotations(frame_annotations):
    """True when a frame holds at least one model-generated annotation."""
    return any(
        is_ai_generated(annotation)
        for _, annotation in iter_annotations(frame_annotations)
    )


def manual_classes(frame_annotations):
    """Class names a person has labelled on this frame."""
    return {
        class_name
        for class_name, annotation in iter_annotations(frame_annotations)
        if is_manual(annotation)
    }
