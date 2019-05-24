from datetime import datetime

from flask import render_template, flash, redirect, url_for, request, current_app
from flask_login import current_user, login_required

from app.main import bp
from app import db
from app.main.forms import EditProfileForm, AcquisitionForm
from app.models import User, Acquisition, ProcessTask


@bp.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()


@bp.route('/', methods=['GET', 'POST'])
@bp.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    form = AcquisitionForm()
    if form.validate_on_submit():
        acquisition = Acquisition(body=form.acquisition.data, author=current_user)
        db.session.add(acquisition)
        db.session.commit()
        flash('Your post is now live!')
        return redirect(url_for('main.index'))

    page = request.args.get('page', 1, type=int)
    acquisitions = current_user.acquisitions.paginate(page, current_app.config['ACQUISITIONS_PER_PAGE'], False)

    next_url = url_for('main.index', page=acquisitions.next_num) \
        if acquisitions.has_next else None
    prev_url = url_for('main.index', page=acquisitions.prev_num) \
        if acquisitions.has_prev else None
    print([x for x in acquisitions.items])

    return render_template('index.html', title='Home', form=form, acquisitions=acquisitions.items, next_url=next_url,
                           prev_url=prev_url)


@bp.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    tasks = ProcessTask.query.all()
    page = request.args.get('page', 1, type=int)

    acquisitions = user.acquisitions.order_by(Acquisition.created_at.desc()).paginate(
        page, current_app.config['ACQUISITIONS_PER_PAGE'], False)

    next_url = url_for('main.user', username=user.username, page=acquisitions.next_num) \
        if acquisitions.has_next else None
    prev_url = url_for('main.user', username=user.username, page=acquisitions.prev_num) \
        if acquisitions.has_prev else None

    return render_template('user.html', user=user, acquisitions=acquisitions.items,
                           next_url=next_url, prev_url=prev_url, tasks=tasks)


@bp.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('main.edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title='Edit Profile', form=form)


@bp.route('/explore')
@login_required
def explore():
    page = request.args.get('page', 1, type=int)
    acquisitions = Acquisition.query.order_by(Acquisition.created_at.desc()).paginate(
        page, current_app.config['ACQUISITIONS_PER_PAGE'], False)

    next_url = url_for('main.explore', page=acquisitions.next_num) \
        if acquisitions.has_next else None
    prev_url = url_for('main.explore', page=acquisitions.prev_num) \
        if acquisitions.has_prev else None

    return render_template("index.html", title='Explore', acquisitions=acquisitions.items, next_url=next_url,
                           prev_url=prev_url)
