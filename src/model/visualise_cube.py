import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import plotly.express as px
import plotly.graph_objects as go


class Cube:
    """
    Plotly visualiser for a three-dimensional constrained phase field.

    Use this class when ``phase_field.constrained_dim == 3`` (e.g. a
    quaternary system with one charge constraint, or any system whose
    constrained space is a 3-D simplex / polyhedron).  For two-dimensional
    constrained spaces use :class:`Square` instead.

    The constructor immediately builds the base figure — corner markers and
    edge lines — by calling :meth:`create_fig_corners`.  All subsequent
    ``plot_*`` methods add traces on top of this base.  Call :meth:`show` (or
    ``show(return_fig=True)``) when you are ready to render.

    Parameters
    ----------
    phase_field : Phase_Field
        Must have ``constrained_dim == 3``, plus ``corners``, ``edges``, and
        ``corner_compositions`` populated (as set by
        ``Phase_Field._find_vertices_and_edges``).
    notebook : bool, optional
        If ``True``, renders with the ``notebook_connected`` Plotly renderer
        (needed inside Jupyter). Default ``False``.

    Raises
    ------
    Exception
        If ``phase_field.constrained_dim != 3``.

    Notes
    -----
    For large grids (many omega points), ``p_mode='nonzero'`` in
    :meth:`plot_prediction_results` can be slow because Plotly must render
    every non-zero scatter point in 3-D.  Prefer ``p_mode='top'`` or
    ``p_mode='surface'`` for interactive performance.
    """

    def __init__(self, phase_field, notebook=False):
        """
        Build the base figure (corners + edges) for a 3-D constrained space.

        Parameters
        ----------
        phase_field : Phase_Field
            Must have ``constrained_dim == 3``, ``corners``, ``edges``, and
            ``corner_compositions``.
        notebook : bool, optional
            Pass ``True`` when running inside a Jupyter notebook so that the
            correct Plotly renderer is selected. Default ``False``.
        """
        d = phase_field.constrained_dim
        if d != 3:
            raise Exception(f'Phase field dimension is {d}, use the Square plotting class for 2D.')
        self.phase_field = phase_field
        self.notebook = notebook
        if hasattr(phase_field, 'precursor_corners'):
            self.create_fig_corners(
                phase_field.convert_to_standard_basis(phase_field.precursor_corners),
                phase_field.precursor_edges,
                phase_field.get_composition_strings(phase_field.precursor_corners)
            )
        else:
            self.create_fig_corners(
                phase_field.corners,
                phase_field.edges,
                phase_field.corner_compositions,
            )

    def create_fig_corners(self, corners, edges, labels, s=10):
        """Build the initial 3-D scatter of corner markers and edge lines."""
        corners = self.phase_field.convert_to_constrained_basis(corners)
        df = pd.DataFrame(corners, columns=['x', 'y', 'z'])
        df['Label'] = labels

        self.fig = px.scatter_3d(
            df, x='x', y='y', z='z', text='Label',
            hover_data={'x': False, 'y': False, 'z': False, 'Label': False},
        )
        self.fig.update_traces(marker={'size': s})

        for edge in edges:
            x = [corners[edge[0]][0], corners[edge[1]][0]]
            y = [corners[edge[0]][1], corners[edge[1]][1]]
            z = [corners[edge[0]][2], corners[edge[1]][2]]
            self.fig = go.Figure(data=self.fig.data + (go.Scatter3d(
                x=x, y=y, z=z, mode='lines',
                line_color='black', hoverinfo='skip', showlegend=False,
            ),))

    def show(self, title=None, show=True, save=None, legend_title=None,
             s=13, return_fig=False):
        """
        Finalise layout and render or export the figure.

        Call this after all ``plot_*`` methods have been called.  Layout
        styling (hidden axes, hover style, legend style) is applied here so
        that it always takes effect last.

        Parameters
        ----------
        title : str, optional
            Figure title text.  If ``None`` no title is set (a title may have
            already been set by :meth:`plot_prediction_results`).
        show : bool, optional
            If ``True`` (default), open the figure in the browser / notebook.
            Set to ``False`` when you only want to save or return the figure.
        save : str or None, optional
            Base filename (without extension) for HTML export.  Pass e.g.
            ``save='my_plot'`` to write ``my_plot.html``.  The file can be
            opened in any browser without a running Python server.
        legend_title : str or None, optional
            Title string rendered above the legend box.
        s : int, optional
            Global font size (Arial) for all text in the figure. Default 13.
        return_fig : bool, optional
            If ``True``, return the :class:`plotly.graph_objects.Figure` object
            instead of calling ``fig.show()``.  Useful for embedding the figure
            in a dashboard or post-processing it programmatically.

        Returns
        -------
        plotly.graph_objects.Figure or None
            The figure object when ``return_fig=True``; ``None`` otherwise.
        """
        self.fig.update_coloraxes(showscale=False)
        self.fig.update_scenes(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        self.fig.update_layout(
            font=dict(family='Arial', size=s),
            hoverlabel=dict(
                bgcolor='rgba(25,25,25,0.88)',
                font_size=13,
                font_color='white',
                bordercolor='rgba(0,0,0,0)',
            ),
            legend=dict(
                bgcolor='rgba(255,255,255,0.92)',
                bordercolor='#CCCCCC',
                borderwidth=1,
                title=legend_title,
            ),
        )
        if return_fig:
            return self.fig
        if title is not None:
            self.fig.update_layout(title=dict(text=title))
        if show:
            if self.notebook:
                self.fig.show(renderer='notebook_connected')
            else:
                self.fig.show()
        if save is not None:
            self.fig.write_html(save + '.html')

    def _filter_top_p(self, p, top, omega):
        """Return (omega_top, p_top) for the top `top` fraction of non-zero points by probability."""
        nonzero = p > 0
        p_nz, omega_nz = p[nonzero], omega[nonzero]
        idx = np.argsort(p_nz)[::-1]
        cutoff = max(1, int(np.ceil(top * len(p_nz))))
        return omega_nz[idx[:cutoff]], p_nz[idx[:cutoff]]

    def plot_p(self, p, s=3, omega=None, name=None, opacity=1,
               show_comps_hover=False, comp_precision=2, colourscale='plasma',
               cmin=None, cmax=None, visible=True):
        """
        Plot a scalar probability field as a 3-D scatter coloured by ``p``.

        Typically called internally by :meth:`plot_prediction_results` and
        :meth:`plot_detail_view`, but can also be called directly to overlay
        an arbitrary probability array on the figure.

        For large grids this method renders every supplied point, which can be
        slow in the browser.  Consider filtering with :meth:`plot_p_top` or
        using :meth:`plot_p_surface` for a lightweight surface representation.

        Parameters
        ----------
        p : array-like, shape (N,)
            Scalar values that map to marker colour.  Usually the normalised
            probability density evaluated on ``omega``.
        s : int, optional
            Marker size. Default 3.
        omega : array-like, shape (N, 3), optional
            3-D constrained-basis coordinates of the points.  If ``None``,
            uses ``self.phase_field.omega`` (the standard grid).
        name : str or None, optional
            Legend entry label.  Pass ``None`` to suppress the legend entry.
        opacity : float, optional
            Marker opacity in [0, 1]. Default 1.
        show_comps_hover : bool, optional
            If ``True``, include the composition string in the hover tooltip.
            Slightly slower for large grids. Default ``False``.
        comp_precision : int, optional
            Decimal places in composition strings when ``show_comps_hover`` is
            ``True``. Default 2.
        colourscale : str, optional
            Plotly colourscale name. Default ``'plasma'``.
        cmin : float or None, optional
            Lower bound of the colour axis.  ``None`` uses the data minimum.
        cmax : float or None, optional
            Upper bound of the colour axis.  ``None`` uses the data maximum.
        visible : bool or str, optional
            Plotly ``visible`` property.  ``True`` (default) shows immediately;
            ``'legendonly'`` hides the trace but keeps the legend toggle.
        """
        if omega is None:
            omega = self.phase_field.omega

        if show_comps_hover:
            comp_strings = self.phase_field.get_composition_strings(omega, prec=comp_precision)
            hovertemplate = '<b>%{customdata[0]}</b><br>p=%{marker.color:.2e}<extra></extra>'
            customdata = [[c] for c in comp_strings]
        else:
            hovertemplate = 'p=%{marker.color:.2e}<extra></extra>'
            customdata = None

        trace = go.Scatter3d(
            x=omega[:, 0], y=omega[:, 1], z=omega[:, 2],
            mode='markers',
            marker=dict(
                color=p,
                colorscale=colourscale,
                cmin=cmin, cmax=cmax,
                size=s,
                opacity=opacity,
                showscale=False,
            ),
            name=name if name is not None else '',
            showlegend=name is not None,
            hovertemplate=hovertemplate,
            customdata=customdata,
            visible=visible,
        )
        self.fig.add_trace(trace)

    def plot_p_top(self, p, top=0.9, s=3, omega=None, name=None, opacity=1,
                   show_comps_hover=False, comp_precision=2, colourscale='plasma',
                   cmin=None, cmax=None):
        """
        Plot only the grid points that account for the top fraction of probability mass.

        Convenience wrapper around :meth:`plot_p` that filters ``omega`` and
        ``p`` to the highest-density (non-zero) subset before plotting.  Much
        faster than ``p_mode='nonzero'`` for large grids because it further
        restricts the point count.

        Parameters
        ----------
        p : array-like, shape (N,)
            Probability density values over the full grid.
        top : float, optional
            Fraction of total probability mass to retain, e.g. ``0.9`` keeps
            the 90 % highest-density points. Default ``0.9``.
        s : int, optional
            Marker size. Default 3.
        omega : array-like or None, optional
            Grid coordinates.  ``None`` uses ``self.phase_field.omega``.
        name : str or None, optional
            Legend label.
        opacity : float, optional
            Marker opacity. Default 1.
        show_comps_hover : bool, optional
            Show composition strings in hover tooltip. Default ``False``.
        comp_precision : int, optional
            Decimal places in composition strings. Default 2.
        colourscale : str, optional
            Plotly colourscale name. Default ``'plasma'``.
        cmin : float or None, optional
            Lower bound of the colour axis.
        cmax : float or None, optional
            Upper bound of the colour axis.
        """
        if omega is None:
            omega = self.phase_field.omega
        omega_top, p_top = self._filter_top_p(p, top, omega)
        self.plot_p(p_top, s=s, omega=omega_top, name=name, opacity=opacity,
                    show_comps_hover=show_comps_hover, comp_precision=comp_precision,
                    colourscale=colourscale, cmin=cmin, cmax=cmax)

    def plot_p_surface(self, p, top=0.9, omega=None, name=None,
                       colourscale='plasma', opacity=0.35):
        """
        Plot a convex-hull isosurface enclosing the top probability-mass region.

        Computes a :class:`scipy.spatial.ConvexHull` over the highest-density
        ``top`` fraction of non-zero grid points, then renders it as a
        semi-transparent :class:`plotly.graph_objects.Mesh3d` coloured by
        vertex probability.  This is the lightest-weight 3-D density
        visualisation and is the default for ``p_mode='surface'`` in
        :meth:`plot_prediction_results`.

        Parameters
        ----------
        p : array-like, shape (N,)
            Probability density values over the full grid.
        top : float, optional
            Fraction of total probability mass whose enclosing convex hull is
            drawn.  Default ``0.9`` (90 % highest-density region).
        omega : array-like or None, optional
            Grid coordinates.  ``None`` uses ``self.phase_field.omega``.
        name : str or None, optional
            Legend label for the mesh trace.
        colourscale : str, optional
            Plotly colourscale applied to the mesh vertex intensities.
            Default ``'plasma'``.
        opacity : float, optional
            Mesh surface opacity in [0, 1]. Default ``0.35`` — semi-transparent
            so internal structures remain visible.

        Raises
        ------
        Exception
            If the top region contains fewer than 4 non-coplanar points (the
            minimum required to construct a 3-D convex hull).
        """
        if omega is None:
            omega = self.phase_field.omega
        omega_top, p_top = self._filter_top_p(p, top, omega)

        if len(omega_top) < 4:
            raise Exception(
                f'Only {len(omega_top)} points in top {top*100:.0f}% — '
                'need at least 4 for a 3D convex hull.')

        hull = ConvexHull(omega_top)
        i, j, k = hull.simplices[:, 0], hull.simplices[:, 1], hull.simplices[:, 2]

        mesh = go.Mesh3d(
            x=omega_top[:, 0], y=omega_top[:, 1], z=omega_top[:, 2],
            i=i, j=j, k=k,
            intensity=p_top,
            colorscale=colourscale,
            opacity=opacity,
            hoverinfo='skip',
            showscale=False,
        )
        if name is not None:
            mesh.update(name=name, showlegend=True)

        self.fig.add_trace(mesh)

    def plot_points(self, points, color=None, name=None, legend_on_hover=True,
                   norm=None, custom_labels=None, s=6, tol=1e-1,
                   marker=None, border_color=None, border_width=1.5):
        """
        Plot one or more composition points as a 3-D scatter trace.

        Accepts coordinates in either the standard (element-fraction) basis or
        the 3-D constrained basis.  The method validates charge neutrality,
        generates composition-string hover labels, and adds a single
        :class:`plotly.graph_objects.Scatter3d` trace.

        Call directly when you want to overlay arbitrary compositions that are
        not part of a :class:`Prediction` (e.g. candidate targets, literature
        values).

        Parameters
        ----------
        points : array-like, shape (N, 3) or (N, n_elements)
            Composition coordinates.  If the second dimension equals
            ``phase_field.nelements`` the points are automatically converted
            to the constrained basis.
        color : str or None, optional
            Marker fill colour (any Plotly colour string).  ``None`` uses the
            Plotly default.
        name : str or None, optional
            Legend entry label.  ``None`` suppresses the legend entry.
        legend_on_hover : bool, optional
            Unused; kept for API backwards compatibility.
        norm : float or None, optional
            Passed to ``phase_field.get_composition_strings`` to normalise the
            displayed fractions.  ``None`` uses the raw constrained-basis values.
        custom_labels : list of str or None, optional
            HTML strings prepended to each point's hover block (before the
            composition string).  Use ``<b>...</b>`` for bold text.  Must have
            the same length as ``points``.
        s : int, optional
            Marker size in pixels. Default 6.
        tol : float, optional
            Tolerance passed to ``phase_field.check_neutrality``. Default 1e-1.
        marker : str or None, optional
            Plotly marker symbol (e.g. ``'diamond'``, ``'cross'``).  ``None``
            uses the default circle.
        border_color : str or None, optional
            Marker border colour.  ``None`` means no border.
        border_width : float, optional
            Marker border width. Default 1.5.

        Raises
        ------
        Exception
            If any point is not charge neutral within ``tol``, or if the point
            array has an unexpected shape.
        """
        points = np.array(points)
        if points.ndim == 1:
            points = np.array([points])
        if points.shape[1] == self.phase_field.nelements:
            points = self.phase_field.convert_to_constrained_basis(points)
        if points.shape[1] != self.phase_field.constrained_dim:
            raise Exception('Points must be provided in standard or constrained basis.')
        if points.shape[1] != 3:
            raise Exception('Constrained basis must be 3D.')

        points_s = self.phase_field.convert_to_standard_basis(points)
        self.phase_field.check_neutrality(points_s)

        formulas = (self.phase_field.get_composition_strings(points) if norm is None
                    else self.phase_field.get_composition_strings(points, norm=norm))
        if isinstance(formulas, str):
            formulas = [formulas]

        # Clean hovertemplate — no auto column-name prefixes
        hover_cols = []
        hover_parts = []
        if custom_labels is not None:
            hover_cols.append(list(custom_labels))
            hover_parts.append('%{customdata[0]}')
        hover_cols.append(list(formulas))
        hover_parts.append(f'%{{customdata[{len(hover_cols) - 1}]}}')

        customdata = [list(row) for row in zip(*hover_cols)]
        hovertemplate = '<br>'.join(hover_parts) + '<extra></extra>'

        marker_dict = dict(size=s)
        if color is not None:
            marker_dict['color'] = color
        if marker is not None:
            marker_dict['symbol'] = marker
        if border_color is not None:
            marker_dict['line'] = dict(color=border_color, width=border_width)

        trace = go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            name=name,
            showlegend=name is not None,
            customdata=customdata,
            hovertemplate=hovertemplate,
            marker=marker_dict,
        )
        self.fig.add_trace(trace)

    def plot_prediction_results(
            self, prediction, grid_s=3, point_s=6, show_comps_hover=True,
            comp_precision=2, colourscale='plasma', plot_samples=True,
            plot_knowns=True, plot_average_known=False,
            p_mode='nonzero', top=0.9, surface_opacity=0.35, plot_individual=True,
            visible=True):
        """
        Plot the main PICIP result: probability density over the 3-D phase
        field with optional sample, known, and suggestion overlays.

        This is the primary method for visualising a :class:`Prediction` object
        in three dimensions.  Traces are added in this order: probability
        density (controlled by ``p_mode``), per-sample individual densities
        (legend-only), sample points, known compositions, average-known marker,
        suggestion points.

        Parameters
        ----------
        prediction : Prediction
            Result returned by ``PICIP.run()``.
        grid_s : int, optional
            Marker size for scatter-based density modes (``'all'``,
            ``'nonzero'``, ``'top'``). Default 3.
        point_s : int, optional
            Marker size for sample / known / suggestion scatter points. Default 6.
        show_comps_hover : bool, optional
            If ``True`` (default), show the composition string in the hover
            tooltip for each density point.
        comp_precision : int, optional
            Number of decimal places used in composition strings. Default 2.
        colourscale : str, optional
            Plotly colourscale name for the probability density. Default
            ``'plasma'``.
        plot_samples : bool, optional
            Overlay the measured sample composition(s) as white circle markers.
            Default ``True``.
        plot_knowns : bool, optional
            Overlay the known co-existing phase compositions as amber diamond
            markers (deduplicated across samples). Default ``True``.
        plot_average_known : bool, optional
            Overlay the weighted-average known composition (the centroid used
            internally by the algorithm) as an amber cross marker. Useful for
            debugging. Default ``False``.
        p_mode : {'nonzero', 'all', 'top', 'surface'}, optional
            Controls how the probability density is rendered:

            * ``'nonzero'`` — scatter of all grid points with ``p > 0``.
              Can be slow for large grids; use ``'top'`` or ``'surface'``
              for better interactive performance.
            * ``'all'`` — scatter of every grid point including zero-density
              points (rarely useful; included for completeness).
            * ``'top'`` — scatter of the ``top`` highest-density fraction of
              non-zero points only.
            * ``'surface'`` — semi-transparent convex-hull isosurface
              enclosing the ``top`` highest-density region, rendered via
              :meth:`plot_p_surface`.  Lightest-weight option for large grids.

            Default ``'nonzero'``.
        top : float, optional
            Fraction of probability mass used by ``'top'`` and ``'surface'``
            modes. Default ``0.9``.
        surface_opacity : float, optional
            Opacity of the convex-hull surface when ``p_mode='surface'``.
            Default ``0.35``.
        plot_individual : bool, optional
            When ``prediction`` combines multiple samples, also add each
            sample's individual density as a ``'legendonly'`` trace.
            Default ``True``.
        visible : bool or str, optional
            Plotly ``visible`` property for the primary density trace.
            ``True`` (default) shows it immediately; ``'legendonly'`` hides it
            on load but keeps the legend toggle.

        Raises
        ------
        Exception
            If ``p_mode`` is not one of the recognised options.

        Notes
        -----
        The figure title is generated automatically from the prediction's
        sample names, compositions, and ``predicted_error`` values, and is
        only set when ``visible is True`` to avoid overwriting a title set by
        a previous call.
        """
        # Default title (only set on the first call per figure — don't overwrite)
        if visible is True:
            samples = prediction.samples
            if len(samples) == 1:
                s = samples[0]
                knowns_str = ' + '.join(
                    f'{k.reduced_formula} ({w:.0%})'
                    for k, w in zip(s.knowns, s.weights)
                )
                title = f'{s.name} | {s.composition.reduced_formula} | {knowns_str} | σ={s.predicted_error}'
            else:
                parts = [f'{s.name}: {s.composition.reduced_formula} (σ={s.predicted_error})' for s in samples]
                title = 'combined | ' + ' | '.join(parts)
            self.fig.update_layout(title=dict(text=title))

        p_name = f'probability ({prediction.name})'
        if p_mode == 'all':
            self.plot_p(prediction.prob_density, s=grid_s, name=p_name,
                        show_comps_hover=show_comps_hover, comp_precision=comp_precision,
                        colourscale=colourscale, visible=visible)
        elif p_mode == 'nonzero':
            p = prediction.prob_density
            mask = p > 0
            self.plot_p(p[mask], s=grid_s, omega=self.phase_field.omega[mask],
                        name=p_name, show_comps_hover=show_comps_hover,
                        comp_precision=comp_precision, colourscale=colourscale,
                        visible=visible)
        elif p_mode == 'top':
            self.plot_p_top(prediction.prob_density, top=top, s=grid_s,
                            name=p_name, show_comps_hover=show_comps_hover,
                            comp_precision=comp_precision, colourscale=colourscale)
            if visible != True:
                self.fig.data[-1].update(visible=visible)
        elif p_mode == 'surface':
            self.plot_p_surface(prediction.prob_density, top=top,
                                name=p_name, colourscale=colourscale,
                                opacity=surface_opacity)
            if visible != True:
                self.fig.data[-1].update(visible=visible)
        else:
            raise Exception(f"Unknown p_mode '{p_mode}'. Use 'all', 'nonzero', 'top', or 'surface'.")

        if plot_individual and prediction.individual_densities is not None:
            for samp, p_ind in zip(prediction.samples, prediction.individual_densities):
                ind_name = f'probability ({samp.name})'
                if p_mode == 'top':
                    self.plot_p_top(p_ind, top=top, s=grid_s, name=ind_name,
                                    colourscale=colourscale)
                elif p_mode == 'surface':
                    self.plot_p_surface(p_ind, top=top, name=ind_name,
                                        colourscale=colourscale, opacity=surface_opacity)
                else:
                    mask = p_ind > 0
                    self.plot_p(p_ind[mask], s=grid_s, omega=self.phase_field.omega[mask],
                                name=ind_name, colourscale=colourscale)
                # hide from view on load — toggle in legend
                self.fig.data[-1].update(visible='legendonly')

        if plot_samples:
            multi = len(prediction.samples) > 1
            samples_c, custom_labels = [], []
            for samp in prediction.samples:
                samples_c.append(samp.comp_constrained)
                if plot_knowns:
                    known_strs = [k.reduced_formula for k in samp.knowns]
                    custom_labels.append(f"<b>{samp.name}</b><br>knowns: {', '.join(known_strs)}")
                else:
                    custom_labels.append(f"<b>{samp.name}</b>")
            legend_name = 'samples' if multi else prediction.samples[0].name
            self.plot_points(samples_c, color='white', name=legend_name,
                             custom_labels=custom_labels, s=point_s,
                             border_color='#222222', border_width=2)

            if plot_knowns:
                seen = {}
                for samp in prediction.samples:
                    for k_comp, k_c in zip(samp.knowns, samp.knowns_constrained):
                        key = k_comp.reduced_formula
                        if key not in seen:
                            seen[key] = k_c
                if seen:
                    known_coords = np.array(list(seen.values()))
                    known_labels = [f"<b>{k}</b>" for k in seen.keys()]
                    self.plot_points(known_coords, color='#E8A838', name='knowns',
                                     custom_labels=known_labels, s=point_s,
                                     marker='diamond', border_color='#222222', border_width=2)

            if plot_average_known:
                avg_k_coords = np.array([s.average_k for s in prediction.samples])
                avg_k_labels = [f"<b>{s.name}</b><br>avg. known" for s in prediction.samples]
                self.plot_points(avg_k_coords, color='#E8A838', name='avg. known',
                                 custom_labels=avg_k_labels, s=point_s,
                                 marker='cross', border_color='#222222', border_width=2)

            if prediction.suggestions is not None:
                self.plot_suggestions(prediction.suggestions)

    def plot_detail_view(self, prediction,
                         show_projected_scatter=True, n_lines=None,
                         line_color='rgba(160,160,160,0.4)', line_width=1,
                         simplex_s=5, scatter_s=2, colourscale='plasma',
                         top_mode='surface', top=0.9, surface_opacity=0.35):
        """
        Detailed diagnostic view of the algorithm internals for a single sample.

        Requires a single-sample :class:`Prediction` (i.e. produced by
        ``PICIP.run(sample_index=i)``).  Layers are added from bottom to top:

        1. **Support assumption** — Dirichlet PDF evaluated on the known
           simplex triangle (skipped for single-known samples that use the
           Gaussian path).
        2. **Projected distribution** — raw projected points from the simplex
           coloured by probability weight (semi-transparent).
        3. **Interpolated distribution** — the final ``griddata`` result on
           the omega grid (non-zero points only, semi-transparent).
        4. **Top overlay** — either a convex-hull surface (``top_mode='surface'``,
           default) or a scatter of the top-fraction points
           (``top_mode='top'``), or nothing (``top_mode=None``).
        5. **Known compositions** — amber diamond markers.
        6. **Average known** — amber cross marker.
        7. **Sample composition** — white circle marker.
        8. **Suggestions** — if ``prediction.suggestions`` is set.

        Parameters
        ----------
        prediction : Prediction
            Must contain exactly one sample.  Raises ``ValueError`` if more
            than one sample is present.
        show_projected_scatter : bool, optional
            Include the projected-distribution scatter layer. Default ``True``.
        n_lines : int or None, optional
            Number of projected ray lines to draw.  ``None`` draws all rays,
            which can be slow for high ``n_l`` values. Default ``None``.
        line_color : str, optional
            RGBA colour string for ray lines. Default
            ``'rgba(160,160,160,0.4)'``.
        line_width : float, optional
            Width of the ray lines. Default 1.
        simplex_s : int, optional
            Marker size for the support-assumption (simplex) layer. Default 5.
        scatter_s : int, optional
            Marker size for the projected-distribution and interpolated-
            distribution scatter layers. Default 2.
        colourscale : str, optional
            Plotly colourscale applied to all probability heatmap layers.
            Default ``'plasma'``.
        top_mode : {'surface', 'top', None}, optional
            Controls the top-overlay layer:

            * ``'surface'`` (default) — semi-transparent convex-hull surface
              via :meth:`plot_p_surface`.
            * ``'top'`` — scatter points for the top-fraction region via
              :meth:`plot_p_top`.
            * ``None`` — no overlay.
        top : float, optional
            Fraction of probability mass for the top-overlay. Default ``0.9``.
        surface_opacity : float, optional
            Opacity of the convex-hull surface when ``top_mode='surface'``.
            Default ``0.35``.

        Raises
        ------
        ValueError
            If ``prediction`` contains more than one sample.
        Exception
            If the sample has no ``_simplex_points`` attribute (i.e. PICIP
            has not been run yet), or if ``top_mode`` is not recognised.
        """
        n_samples = len(prediction.samples)
        if n_samples > 1:
            raise ValueError(
                f"plot_detail_view requires a single-sample prediction "
                f"({n_samples} samples given). Run PICIP.run(sample_index) "
                "to get a prediction for one sample."
            )
        sample_index = 0

        sample = prediction.samples[sample_index]
        if not hasattr(sample, '_simplex_points'):
            raise Exception(
                'No detail data on this sample. Run PICIP.run() to populate it.')

        knowns_str = ' + '.join(
            f'{k.reduced_formula} ({w:.0%})'
            for k, w in zip(sample.knowns, sample.weights)
        )
        pf = self.phase_field
        subtitle = (f'ω: {len(pf.omega)} pts | '
                    f'n_l={sample._n_rays}, n_p={sample._n_p}')
        title = (f'{sample.name} | {sample.composition.reduced_formula} | '
                 f'{knowns_str} | σ={sample.predicted_error}'
                 f'<br><sup>{subtitle}</sup>')
        self.fig.update_layout(title=dict(text=title))

        # 1 — Support assumption: Dirichlet PDF on the known simplex (skipped for single-known)
        if sample._simplex_points is not None:
            p = sample._simplex_pdf
            self.plot_p(p, omega=sample._simplex_points,
                        name='support assumption', s=simplex_s, colourscale=colourscale,
                        cmin=0, cmax=p.max())

        # 2 — Projected distribution: raw projected points coloured by probability
        if show_projected_scatter:
            p = sample._projected_p
            self.plot_p(p, omega=sample._projected_points,
                        name='projected distribution', s=scatter_s, colourscale=colourscale,
                        opacity=0.5, cmin=0, cmax=p.max())

        # 3 — Interpolated distribution: griddata result on omega (nonzero only)
        p = prediction.prob_density
        mask = p > 0
        self.plot_p(p[mask], omega=self.phase_field.omega[mask],
                    name='interpolated distribution', s=scatter_s, colourscale=colourscale,
                    opacity=0.3, cmin=0, cmax=p[mask].max())

        # 4 — Top-x% overlay (surface by default)
        if top_mode == 'top':
            self.plot_p_top(prediction.prob_density, top=top,
                            name=f'top {int(top*100)}%', colourscale=colourscale)
        elif top_mode == 'surface':
            self.plot_p_surface(prediction.prob_density, top=top,
                                name=f'top {int(top*100)}% surface',
                                colourscale=colourscale, opacity=surface_opacity)
        elif top_mode is not None:
            raise Exception(f"top_mode '{top_mode}' not recognised. Use 'top', 'surface', or None.")

        # 5 — Known compositions
        known_labels = [f'<b>{str(k.reduced_formula)}</b>' for k in sample.knowns]
        self.plot_points(sample.knowns_constrained, color='#E8A838', name='knowns',
                         custom_labels=known_labels, s=6, marker='diamond',
                         border_color='#222222', border_width=2)

        # 6 — Average known
        self.plot_points([sample.average_k], color='#E8A838', name='avg. known',
                         custom_labels=['<b>avg. known</b>'], s=6,
                         marker='cross', border_color='#222222', border_width=2)

        # 7 — Sample composition
        self.plot_points([sample.comp_constrained], color='white', name=sample.name,
                         custom_labels=[f'<b>{sample.name}</b>'], s=6,
                         border_color='#222222', border_width=2)

        if prediction.suggestions is not None:
            self.plot_suggestions(prediction.suggestions)

    def plot_mesh(self, points, name=None, as_hull=True):
        """
        Plot a 3-D mesh from an array of points.

        Parameters
        ----------
        points : array-like, shape (N, 3)
            Coordinates of the mesh vertices in the constrained basis.
        name : str or None, optional
            Legend label. Default ``None``.
        as_hull : bool, optional
            If ``True`` (default), compute the convex hull of ``points`` and
            render the hull facets as a ``Mesh3d`` surface.  If ``False``,
            render the raw points as a 3-D scatter instead.
        """
        points = np.array(points)
        if as_hull:
            hull = ConvexHull(points)
            i, j, k = hull.simplices[:, 0], hull.simplices[:, 1], hull.simplices[:, 2]
            fig = go.Figure(data=[go.Mesh3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                i=i, j=j, k=k, color='lightpink', opacity=0.4,
            )])
        else:
            fig = px.scatter_3d(x=points[:, 0], y=points[:, 1], z=points[:, 2])
        if name is not None:
            fig.update_traces(showlegend=True, name=name)
        for trace in fig.data:
            self.fig.add_trace(trace)

    def plot_line(self, start, end, c='black', w=2, name=None):
        """
        Plot a straight line between two points in 3-D.

        Parameters
        ----------
        start : array-like, length 3
            Starting point (x, y, z) in the constrained basis.
        end : array-like, length 3
            Ending point (x, y, z) in the constrained basis.
        c : str, optional
            Line colour. Default ``'black'``.
        w : float, optional
            Line width. Default 2.
        name : str or None, optional
            Legend label. ``None`` suppresses the legend entry.
        """
        trace = go.Scatter3d(
            x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
            mode='lines', line=dict(color=c, width=w),
        )
        if name is not None:
            trace.update(name=name, showlegend=True)
        else:
            trace.update(showlegend=False)  # type: ignore
        self.fig.add_trace(trace)

    def plot_suggestions(self, suggestions, s=6):
        """
        Overlay next-experiment suggestion points on the 3-D figure.

        Styled according to the label stored in ``suggestions.labels``:

        * ``'mean'`` — white fill, amber (``#E8A838``) border; represents the
          posterior mean of the prediction.
        * ``'sampled'`` — amber fill, dark border; represents points drawn
          randomly from the posterior distribution.

        Called automatically by :meth:`plot_prediction_results` and
        :meth:`plot_detail_view` when ``prediction.suggestions`` is not
        ``None``.  Call directly if you want to overlay suggestions on a
        figure built manually with :meth:`plot_p` or :meth:`plot_points`.

        Parameters
        ----------
        suggestions : Suggestions
            Object returned by ``PICIP.suggest()``.  Must have
            ``constrained``, ``labels``, and ``elements`` attributes.
        s : int, optional
            Marker size. Default 6.
        """
        labels = np.array(suggestions.labels)

        mean_pts = suggestions.constrained[labels == 'mean']
        if len(mean_pts):
            self.plot_points(mean_pts, color='white', name='suggestion (mean)',
                             custom_labels=['<b>mean</b>'] * len(mean_pts),
                             s=s, marker='circle',
                             border_color='#E8A838', border_width=3)

        samp_pts = suggestions.constrained[labels == 'sampled']
        if len(samp_pts):
            samp_labels = [f'<b>sampled {i+1}</b>' for i in range(len(samp_pts))]
            self.plot_points(samp_pts, color='#E8A838', name='suggestions (sampled)',
                             custom_labels=samp_labels, s=s, marker='circle',
                             border_color='#222222', border_width=2)


def make_plotter(phase_field):
    """Return the correct plotter for a phase field — Square (2-D) or Cube (3-D).

    Selects the visualiser class based on ``phase_field.constrained_dim``, so
    you do not need to track which class to import after precursors have
    potentially reduced the dimension.

    Parameters
    ----------
    phase_field : Phase_Field
        A fully set-up phase field (any ``setup_*`` method).

    Returns
    -------
    Square or Cube
        A freshly constructed plotter ready for ``plot_prediction_results`` etc.

    Raises
    ------
    ValueError
        If ``constrained_dim`` is not 2 or 3.

    Examples
    --------
    >>> pl = make_plotter(pf)
    >>> pl.plot_prediction_results(pred)
    >>> pl.show()
    """
    from .visualise_square import Square
    dim = phase_field.constrained_dim
    if dim == 2:
        return Square(phase_field)
    if dim == 3:
        return Cube(phase_field)
    raise ValueError(
        f"No plotter available for constrained_dim={dim}. "
        "Expected 2 (Square) or 3 (Cube)."
    )

