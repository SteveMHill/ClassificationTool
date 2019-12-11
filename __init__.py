# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ClassificationTool
                                 A QGIS plugin
 Classification of remote sensing images
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2019-05-24
        copyright            : (C) 2019 by Steven Hill
        email                : steven.hill@uni-wuerzburg.de
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


def classFactory(iface):  # pylint: disable=invalid-name

    from .ClassificationTool import ClassificationTool

    return ClassificationTool(iface)
