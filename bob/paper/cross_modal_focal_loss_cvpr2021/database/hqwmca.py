#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import bob.io.base
from bob.pad.base.database import PadDatabase, PadFile
from bob.extension import rc
from bob.paper.cross_modal_focal_loss_cvpr2021.preprocessor.FaceCropAlign import detect_face_landmarks_in_image
from bob.paper.cross_modal_focal_loss_cvpr2021.preprocessor import FaceCropAlign, VideoFaceCropAlignBlockPatch


#  (
#     ,
# )
# from bob.db.hqwmca.preprocessor.FaceCropAlign import detect_face_landmarks_in_image

from bob.db.hqwmca.attack_dictionaries import (
	idiap_type_id_config,
	idiap_subtype_id_config,
)
from bob.io.stream import Stream, StreamFile

import cv2
import bob.io.image
from bob.bio.video.annotator import Wrapper


class HQWMCAPadFile(PadFile):

	"""
	A high level implementation of the File class for the HQWMCA database.

	Attributes
	----------
	vf : :py:class:`object`
	An instance of the VideoFile class defined in the low level db interface
	of the HQWMCA database, in the bob.db.hqwmca.models.py file.
	streams: :py:dict:
	Dictionary of bob.io.stream Stream objects. Should be defined in a configuration file

	"""

	def __init__(self, vf, streams=None, n_frames=10, **kwargs):

		""" Init

		Parameters
		----------
		vf : :py:class:`object`
			An instance of the VideoFile class defined in the low level db interface
			of the HQWMCA database, in the bob.db.hqwmca.models.py file.
		streams: :py:dict:
			Dictionary of bob.io.stream Stream objects. Should be defined in a configuration file
		n_frames: int:
			The number of frames, evenly spread, you would like to retrieve 
		
		"""

		self.vf = vf

		# print('.................=============== self.annotation_directory FILE', self.annotation_directory)
		attack_type = str(vf.type_id)

		if vf.is_attack():
			pai_desc = idiap_type_id_config[str(vf.type_id)]
			attack_type = "attack/" + pai_desc
		else:
			attack_type = None

		super(HQWMCAPadFile, self).__init__(
			client_id=vf.client_id,
			file_id=vf.id,
			attack_type=attack_type,
			path=vf.path,
			**kwargs
		)

		self.streams = streams
		self.n_frames = n_frames

	def load(self, directory=rc["bob.db.hqwmca.directory"], extension=".h5", streams=None, n_frames=None):
		""" Loads data from the given file

		Parameters
		----------
		directory : :py:class:`str`
			String containing the path to the HQWMCA database 
		extension : :py:class:`str`
			Typical extension of a VideoFile
		streams: : :py:class:`dict`
			A dictionary with keys stream names, and as values :py:class`bob.io.stream.Stream` objects to load the data.
			Default is None, in which case self.streams is used.
		n_frames: :py:class:`int`
			Number of evenly spreaded frames to load in the streams. Default is None, in which case self.n_frames is used.

		Returns
		-------
		dict:
			Dictionary where the keys are the stream names and value is the streams data (VideoLikeContainer).
		"""

		if streams is None:
			streams = self.streams
		if n_frames is None:
			n_frames = self.n_frames

		return self.vf.load(
			directory, extension, streams=streams, n_frames=n_frames
		)

	@property
	def annotations(self):

		"""
		Computes annotations for this BatlCSVPadFile.

		NOTE: you can pre-compute annotation in your first experiment
		and then reuse them in other experiments setting
		``self.annotations_temp_dir`` path of the BatlCSVPadDatabase class, 
		where precomputed annotations will be saved.

		**Returns:**

		``annotations`` : :py:class:`dict`
				A dictionary containing annotations for
				each frame in the video.
				Dictionary structure:
				``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
				Where
				``frameN_dict`` contains coordinates of the
				face bounding box and landmarks in frame N.
		"""

		file_path = os.path.join(
			self.annotation_directory, self.vf.path + ".json"
		)

		if not os.path.isfile(file_path):  # no file with annotations

			video_dict = self.load(streams={'color': self.streams['color']})

			annotator = bob.bio.video.annotator.Wrapper(
				"mtcnn", normalize=False
			)

			annotations = annotator([video_dict['color']])

			print('annotations', type(annotations), annotations)

			if self.annotation_directory:  # if directory is not an empty string

				bob.io.base.create_directories_safe(
					directory=os.path.split(file_path)[0], dryrun=False
				)

				with open(file_path, "w+") as json_file:
					json_file.write(json.dumps(annotations))

		else:  # if file with annotations exists load them from file

			try:

				with open(file_path, "r") as json_file:

					annotations = json.load(json_file)
			except Exception as e:
				print(e)
				annotations=None

		if not annotations:  # if dictionary is empty

			return None

		return annotations[0]


class HQWMCAPadDatabase(PadDatabase):
	"""High level implementation of the Database class for the HQWMCA database.
 
	Attributes
	----------
	db : :py:class:`bob.db.hqwmca.Database`
		the low-level database interface
	streams: :py:dict:
		Dictionary of bob.io.stream Stream objects. Should be defined in a configuration file

	"""

	def __init__(
		self,
		protocol="grand_test",
		original_directory=rc["bob.db.hqwmca.directory"],
		original_extension=".h5",
		annotations_dir=None,
		streams=None,
		n_frames=10,
		use_curated_file_list=False,
		**kwargs
	):
		"""Init function

			Parameters
			----------
			protocol : :py:class:`str`
				The name of the protocol that defines the default experimental setup for this database.
			original_directory : :py:class:`str`
				The directory where the original data of the database are stored.
			original_extension : :py:class:`str`
				The file name extension of the original data.
			annotations_dir: str
				Path to the annotations
			streams: :py:dict:
				Dictionary of bob.io.stream Stream objects. Should be defined in a configuration file
			n_frames: int:
				The number of frames, evenly spread, you would like to retrieve 
			use_curated_file_list: bool
				Whether to remove all light makeup, unisex glasses and wigs, which are border case attacks, to create a clean set of attacks
				Removes these attacks from all folds. This can either be set as argument or as additional '-curated' in the protocol name.
			
		"""
		from bob.db.hqwmca import Database as LowLevelDatabase

		self.db = LowLevelDatabase()
		self.streams = streams
		self.n_frames = n_frames
		self.annotations_dir = annotations_dir
		self.use_curated_file_list = use_curated_file_list

		print(
			"...............................HLDI self.annotation_directory",
			self.annotations_dir,
		)

		super(HQWMCAPadDatabase, self).__init__(
			name="hqwmca",
			protocol=protocol,
			original_directory=original_directory,
			original_extension=original_extension,
		)

		self.low_level_group_names = ("train", "validation", "test")
		self.high_level_group_names = ("train", "dev", "eval")

	@property
	def original_directory(self):
		return self.db.original_directory

	@original_directory.setter
	def original_directory(self, value):
		self.db.original_directory = value
	def unseen_attack_list_maker(self,files,unknown_attack,train=True):
		"""
		Selects and returns a list of files for Leave One Out (LOO) protocols. 
		This utilizes the original partitioning in the `grandtest` protocol and subselects 
		the file list such that the specified `unknown_attack` is removed from both `train` and `dev` sets. 
		The `test` set will consist of only the selected `unknown_attack` and `bonafide` files.

		**Parameters:**

		``files`` : pyclass::list
			A list of files, db.objects()
		``unknown_attack`` : str
			The unknown attack protocol name example:'rigidmask' .
		``train`` : bool
			Denotes whether files are from training or development partition

		**Returns:**

		``mod_files`` : pyclass::list
			A list of files selected for the protocol

		"""


		mod_files=[]


		for file in files:

			if file.is_attack():
			  attack_category = idiap_type_id_config[str(file.type_id)]
			else:
			  attack_category ='bonafide'

			if train:
				if  attack_category==unknown_attack:
					pass
				else:
					mod_files.append(file) # everything except the attack specified is there 

			if not train:

				if  attack_category==unknown_attack or attack_category=='bonafide':
					mod_files.append(file) # only the attack mentioned and bonafides in testing
				else:
					pass


		return mod_files


	def objects(
		self,
		groups=None,
		protocol=None,
		purposes=None,
		model_ids=None,
		attack_types=None,
		**kwargs
	):
		"""Returns a list of HQWMCAPadFile objects, which fulfill the given restrictions.

		Parameters
		----------
		groups : list of :py:class:`str`
			The groups of which the clients should be returned.
			Usually, groups are one or more elements of ('train', 'dev', 'eval')
		protocol : :py:class:`str`
			The protocol for which the samples should be retrieved.
		purposes : :py:class:`str`
			The purposes for which Sample objects should be retrieved.
			Usually it is either 'real' or 'attack'
		model_ids
			This parameter is not supported in PAD databases yet.
		attack_types: list of :py:class:`str`
			The attacks you would like to load.

		Returns
		-------
		samples : :py:class:`HQWMCAPadFile`
				A list of HQWMCAPadFile objects.
		"""

		if groups is None:
			groups = self.high_level_group_names

		if purposes is None:
			purposes = ["real", "attack"]

		groups = self.convert_names_to_lowlevel(
			groups, self.low_level_group_names, self.high_level_group_names
		)

		# if not isinstance(groups, list) and groups is not None and groups is not str:
		#   groups = list(groups)
		if (
			not isinstance(groups, list)
			and groups is not None
			and isinstance(groups, str)
		):  # if a single group is given make it a list
			groups = [groups]

		if (
			len(protocol.split("-")) > 1
			and protocol.split("-")[-1] == "curated"
		):
			self.use_curated_file_list = True

		protocol = protocol.split("-")[0]



		unseen_attack=None

		if 'LOO' in protocol:
		  unseen_attack=protocol.split('_')[-1]
		  self.use_curated_file_list=True
		else:

			files = self.db.objects(
				protocol=protocol,
				groups=groups,
				purposes=purposes,
				attacks=attack_types,
				**kwargs
			)
		if unseen_attack is not None:

		  
			hqwmca_files=[]

			if 'train' in groups:
				t_hqwmca_files = self.db.objects(protocol='grand_test',groups=['train'],purposes=purposes, **kwargs)
				t_hqwmca_files=self.unseen_attack_list_maker(t_hqwmca_files,unseen_attack,train=True)
				hqwmca_files=hqwmca_files+t_hqwmca_files
			if 'validation' in groups: 
				t_hqwmca_files = self.db.objects(protocol='grand_test',groups=['validation'],purposes=purposes, **kwargs)									
				t_hqwmca_files=self.unseen_attack_list_maker(t_hqwmca_files,unseen_attack,train=True)
				hqwmca_files=hqwmca_files+t_hqwmca_files

			if 'test' in groups:
				t_hqwmca_files = self.db.objects(protocol='grand_test',groups=['test'],purposes=purposes, **kwargs)
									  
									  

				t_hqwmca_files=self.unseen_attack_list_maker(t_hqwmca_files,unseen_attack,train=False)

				hqwmca_files=hqwmca_files+t_hqwmca_files

			files=hqwmca_files

		if self.use_curated_file_list:
			# Remove Wigs
			files = [
				f
				for f in files
				if "Wig"
				not in idiap_subtype_id_config[str(f.type_id)][
					str(f.subtype_id)
				]
			]
			# Remove Make up Level 0
			files = [
				f
				for f in files
				if "level 0"
				not in idiap_subtype_id_config[str(f.type_id)][
					str(f.subtype_id)
				]
			]
			# Remove Unisex glasses
			files = [
				f
				for f in files
				if "Unisex glasses"
				not in idiap_subtype_id_config[str(f.type_id)][
					str(f.subtype_id)
				]
			]

		return [
			HQWMCAPadFile(
				f,
				self.streams,
				self.n_frames,
				annotation_directory=self.annotations_dir,
			)
			for f in files
		]

